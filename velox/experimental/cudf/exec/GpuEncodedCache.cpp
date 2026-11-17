/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "velox/experimental/cudf/exec/GpuEncodedCache.h"

#include "velox/common/memory/SharedArbitrator.h"

#include <folly/ScopeGuard.h>

namespace facebook::velox::cudf_velox {

GpuEncodedCache::Completion::~Completion() {
  if (event)
    cudaEventDestroy(event);
}

GpuEncodedCache::Lock::Lock(GpuEncodedCache& cache, Shard& shard)
    : cache_(cache), lock_(shard.mutex, std::defer_lock) {
  const auto start = cache_.profile_ ? now() : 0;
  lock_.lock();
  if (cache_.profile_) {
    acquired_ = now();
    cache_.lockWaitNs_ += acquired_ - start;
  }
}

GpuEncodedCache::Lock::~Lock() {
  if (cache_.profile_)
    cache_.lockHoldNs_ += now() - acquired_;
}

bool GpuEncodedCache::Reclaimer::reclaimableBytes(
    const memory::MemoryPool&,
    uint64_t& bytes) const {
  auto cache = cache_.load().lock();
  bytes = cache ? cache->eligible_.load() : 0;
  return bytes != 0;
}

uint64_t GpuEncodedCache::Reclaimer::reclaim(
    memory::MemoryPool*,
    uint64_t target,
    uint64_t,
    Stats& stats) {
  auto cache = cache_.load().lock();
  if (!cache)
    return 0;
  const auto start = now();
  const auto freed = cache->evict(target);
  stats.reclaimedBytes += freed;
  stats.reclaimExecTimeUs += (now() - start) / 1000;
  return freed;
}

void GpuEncodedCache::Reclaimer::abort(
    memory::MemoryPool*,
    const std::exception_ptr&) {
  if (auto cache = cache_.load().lock()) {
    cache->admissionDisabled_ = true;
    ++cache->aborts_;
    cache->evict(0);
  }
}

void GpuEncodedCache::Lease::prepareCompletion() {
  completion_.emplace_back();
  check(cudaEventCreateWithFlags(
      &completion_.front().event, cudaEventDisableTiming));
}

GpuEncodedCache::Lease::~Lease() {
  if (entry_)
    owner_->release(entry_, stream_, completion_);
}

void GpuEncodedCache::Lease::waitForFillOn(cudaStream_t stream) {
  check(cudaStreamWaitEvent(stream, entry_->ready, 0));
  stream_ = stream;
}

void GpuEncodedCache::Lease::handoffTo(cudaStream_t stream) {
  Completion completion;
  check(cudaEventCreateWithFlags(&completion.event, cudaEventDisableTiming));
  check(cudaEventRecord(completion.event, stream_));
  check(cudaStreamWaitEvent(stream, completion.event, 0));
  stream_ = stream;
}

GpuEncodedCache::GpuEncodedCache(
    std::shared_ptr<memory::CustomMemoryResource> resource,
    cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
    bool profile)
    : resource_(std::move(resource)), profile_(profile) {
  root_ = memory::memoryManager()->addCustomRootPool(
      "gpu-encoded-cache",
      resource_->allocator(),
      resource_->arbitrator(),
      resource_->maxCapacity(),
      std::make_unique<Reclaimer>());
  // Suspension is invoked on the requesting leaf, not the discardable root.
  // Reuse normal Velox driver hooks for fills issued from reader operators.
  leaf_ =
      root_->addLeafChild("payloads", true, exec::MemoryReclaimer::create());
  reporting_.emplace(std::move(upstream), leaf_);
  ensureCudaContextForThread();
  check(cudaStreamCreateWithFlags(&freeStream_, cudaStreamNonBlocking));
}

GpuEncodedCache::~GpuEncodedCache() {
  // A lease retains this owner, so no leased entry can remain at destruction.
  evict(0);
  VELOX_CHECK_EQ(leaf_->usedBytes(), 0);
  cudaStreamDestroy(freeStream_);
}

GpuEncodedCache::LeasePtr GpuEncodedCache::getOrFill(
    const std::string& key,
    uint64_t bytes,
    cudaStream_t stream,
    const std::function<void(void*)>& fill,
    std::shared_ptr<const void> sourceIdentity) {
  return access(key, bytes, stream, &fill, std::move(sourceIdentity));
}

uint64_t GpuEncodedCache::evict(uint64_t target) {
  const auto start = now();
  ensureCudaContextForThread();
  ++calls_;
  requested_ += target;
  // Keep both the map ownership and existing queue nodes until freeing
  // succeeds. No allocation is needed to detach or restore a victim.
  Queue victims;
  uint64_t bytes = 0;
  const auto first = cursor_.fetch_add(1) % kShards;
  try {
    for (size_t i = 0; i < kShards && (!target || bytes < target); ++i) {
      auto& shard = shards_[(first + i) % kShards];
      Lock lock(*this, shard);
      ++shardsVisited_;
      while (!shard.idle.empty() && (!target || bytes < target)) {
        const auto& entry = shard.idle.front();
        VELOX_CHECK_EQ(entry->pins, 0);
        VELOX_CHECK(entry->state == State::Ready);
        entry->state = State::Evicting;
        bytes += entry->bytes;
        eligible_ -= entry->bytes;
        pendingFree_ += entry->bytes;
        victims.splice(victims.end(), shard.idle, shard.idle.begin());
        ++inspected_;
      }
    }
    const auto waitStart = now();
    for (const auto& entry : victims) {
      // A failed acquisition may roll its pin back before a stream wait on
      // readiness was installed. Such an entry still owns a pending fill.
      const auto ready = cudaEventQuery(entry->ready);
      if (ready == cudaErrorNotReady)
        check(cudaEventSynchronize(entry->ready));
      else
        check(ready);
      for (const auto& completion : entry->consumers) {
        const auto status = cudaEventQuery(completion.event);
        if (status == cudaErrorNotReady)
          check(cudaEventSynchronize(completion.event));
        else
          check(status);
      }
    }
    eventWaitNs_ += now() - waitStart;
    const auto freeStart = now();
    {
      // Only allocator calls are serialized. Completion waits above never
      // hold this mutex or a shard lock.
      std::lock_guard lock(freeMutex_);
      while (!victims.empty()) {
        const auto& entry = victims.front();
        reporting_->deallocate(
            cuda::stream_ref{freeStream_},
            entry->data,
            entry->bytes,
            alignof(std::max_align_t));
        pendingFree_ -= entry->bytes;
        freed_ += entry->bytes;
        // The consumer waits succeeded. Event destruction must not throw
        // after freeing the payload, which cannot be reattached anymore.
        cudaEventDestroy(entry->ready);
        entry->consumers.clear();
        auto& shard = shardFor(entry->key);
        {
          Lock shardLock(*this, shard);
          shard.entries.erase(entry->key);
        }
        victims.pop_front();
      }
      leaf_->release();
    }
    freeNs_ += now() - freeStart;
  } catch (...) {
    while (!victims.empty()) {
      const auto& entry = victims.front();
      auto& shard = shardFor(entry->key);
      Lock lock(*this, shard);
      entry->state = State::Ready;
      pendingFree_ -= entry->bytes;
      eligible_ += entry->bytes;
      shard.idle.splice(shard.idle.end(), victims, victims.begin());
    }
    throw;
  }
  reclaimNs_ += now() - start;
  return bytes;
}

folly::dynamic GpuEncodedCache::snapshot() const {
  return folly::dynamic::object("live_bytes", leaf_->usedBytes())(
      "eligible_bytes", eligible_.load())(
      "pending_free_bytes", pendingFree_.load())("hits", hits_.load())(
      "misses", misses_.load())("coalesced_fills", coalesced_.load())(
      "admission_failures", admissionFailures_.load())(
      "fill_errors", fillErrors_.load())("admitted_bytes", admitted_.load())(
      "bypasses", bypasses_.load())("reclaim_calls", calls_.load())(
      "requested_bytes", requested_.load())("freed_bytes", freed_.load())(
      "inspected_entries", inspected_.load())(
      "shards_visited", shardsVisited_.load())(
      "lock_wait_ns", lockWaitNs_.load())("lock_hold_ns", lockHoldNs_.load())(
      "event_wait_ns", eventWaitNs_.load())("free_ns", freeNs_.load())(
      "reclaim_ns", reclaimNs_.load())("cache_aborts", aborts_.load());
}

GpuEncodedCache::LeasePtr GpuEncodedCache::access(
    const std::string& key,
    uint64_t bytes,
    cudaStream_t stream,
    const std::function<void(void*)>* fill,
    std::shared_ptr<const void> sourceIdentity) {
  std::call_once(bindOnce_, [&] {
    static_cast<Reclaimer*>(root_->reclaimer())->bind(weak_from_this());
  });
  ensureCudaContextForThread();
  auto* arbitrator =
      dynamic_cast<memory::SharedArbitrator*>(resource_->arbitrator());
  const bool canAdmit = !admissionDisabled_.load() &&
      (!arbitrator || arbitrator->canAdmitDiscardable());
  Lease lease(shared_from_this(), stream);
  auto& shard = shardFor(key);
  std::shared_ptr<Entry> entry;
  bool creator = false;
  {
    // Waiters pin the filling entry before sleeping. No CUDA operation or
    // device allocator call runs while the shard mutex is held.
    const auto beforeLock = profile_ ? now() : 0;
    std::unique_lock lock(shard.mutex);
    const auto acquired = profile_ ? now() : 0;
    if (profile_)
      lockWaitNs_ += acquired - beforeLock;
    auto it = shard.entries.find(key);
    if (it == shard.entries.end()) {
      if (!fill) {
        if (profile_)
          lockHoldNs_ += now() - acquired;
        return nullptr;
      }
      if (!canAdmit || now() < admitAfterNs_.load()) {
        ++bypasses_;
        if (profile_)
          lockHoldNs_ += now() - acquired;
        return nullptr;
      }
      VELOX_CHECK_GT(bytes, 0);
      entry = std::make_shared<Entry>();
      entry->key = key;
      entry->bytes = bytes;
      entry->sourceIdentity = std::move(sourceIdentity);
      entry->position = shard.pinned.insert(shard.pinned.end(), entry);
      try {
        shard.entries.emplace(key, entry);
      } catch (...) {
        shard.pinned.erase(entry->position);
        throw;
      }
      lease.setEntry(entry);
      creator = true;
      ++misses_;
    } else {
      entry = it->second;
      // A reclaim already owns this allocation. Bypass until it has either
      // freed the entry or restored it after a failed completion wait.
      if (entry->state == State::Evicting) {
        if (fill)
          ++bypasses_;
        return nullptr;
      }
      VELOX_CHECK(
          !bytes || bytes == entry->bytes, "Cache key byte contract changed");
      if (entry->pins++ == 0) {
        shard.pinned.splice(shard.pinned.end(), shard.idle, entry->position);
        eligible_ -= entry->bytes;
      }
      lease.setEntry(entry);
      ++hits_;
      if (entry->state == State::Filling) {
        ++coalesced_;
        if (profile_)
          lockHoldNs_ += now() - acquired;
        lock.unlock();
        // A creator can be suspended by allocation arbitration. Its task must
        // be able to pause while another driver waits for that same fill.
        exec::suspendDriverForMemoryArbitration();
        {
          SCOPE_EXIT {
            exec::resumeDriverAfterMemoryArbitration();
          };
          // Destroy this lock before resuming, which may wait for a task pause.
          std::unique_lock waitLock(shard.mutex);
          entry->filled.wait(
              waitLock, [&] { return entry->state != State::Filling; });
          if (entry->state == State::Failed) {
            --entry->pins;
            lease.setEntry(nullptr);
            return nullptr;
          }
        }
        lease.prepareCompletion();
        check(cudaStreamWaitEvent(stream, entry->ready, 0));
        return std::make_shared<Lease>(std::move(lease));
      }
    }
    if (profile_)
      lockHoldNs_ += now() - acquired;
  }
  if (creator) {
    try {
      lease.prepareCompletion();
      for (unsigned attempt = 0;; ++attempt) {
        try {
          entry->data = reporting_->allocate(
              cuda::stream_ref{stream}, bytes, alignof(std::max_align_t));
          break;
        } catch (...) {
          const auto error = std::current_exception();
          try {
            std::rethrow_exception(error);
          } catch (const std::bad_alloc&) {
          } catch (const VeloxException& exception) {
            if (exception.errorCode() != error_code::kMemCapExceeded &&
                exception.errorCode() != error_code::kMemAllocError) {
              throw;
            }
          }
          // Replace only our idle payloads, at most once, and yield to queries.
          if (attempt != 0 || admissionDisabled_.load() ||
              (arbitrator && !arbitrator->canAdmitDiscardable()) ||
              evict(bytes) == 0) {
            std::rethrow_exception(error);
          }
        }
      }
      check(cudaEventCreateWithFlags(&entry->ready, cudaEventDisableTiming));
      (*fill)(entry->data);
      check(cudaEventRecord(entry->ready, stream));
    } catch (...) {
      const bool submitted = entry->data != nullptr;
      // A partially submitted fill may still touch its destination.
      if (entry->data) {
        cudaStreamSynchronize(stream);
        reporting_->deallocate(
            cuda::stream_ref{stream},
            entry->data,
            bytes,
            alignof(std::max_align_t));
      }
      if (entry->ready)
        cudaEventDestroy(entry->ready);
      {
        Lock lock(*this, shard);
        entry->state = State::Failed;
        shard.entries.erase(key);
        shard.pinned.erase(entry->position);
        --entry->pins;
        lease.setEntry(nullptr);
      }
      entry->filled.notify_all();
      if (!submitted) {
        // Only capacity/allocation refusal is an opportunistic miss. A broken
        // driver hook or other invariant must not silently disable the cache.
        try {
          throw;
        } catch (const std::bad_alloc&) {
        } catch (const VeloxException& error) {
          if (error.errorCode() == error_code::kMemAborted) {
            admissionDisabled_ = true;
          } else if (
              error.errorCode() != error_code::kMemCapExceeded &&
              error.errorCode() != error_code::kMemAllocError) {
            ++fillErrors_;
            throw;
          }
        } catch (...) {
          ++fillErrors_;
          throw;
        }
        ++bypasses_;
        ++admissionFailures_;
        admitAfterNs_ = now() + 1'000'000;
      } else
        ++fillErrors_;
      if (submitted)
        throw; // Do not turn a failed payload write into a cache miss.
      return nullptr;
    }
    {
      Lock lock(*this, shard);
      entry->state = State::Ready;
      admitted_ += bytes;
    }
    entry->filled.notify_all();
  } else {
    lease.prepareCompletion();
  }
  check(cudaStreamWaitEvent(stream, entry->ready, 0));
  return std::make_shared<Lease>(std::move(lease));
}

void GpuEncodedCache::release(
    const std::shared_ptr<Entry>& entry,
    cudaStream_t stream,
    Completions& completion) noexcept {
  ensureCudaContextForThread();
  if (completion.empty() || !completion.front().event ||
      cudaEventRecord(completion.front().event, stream) != cudaSuccess) {
    // Failed lease acquisition or event bookkeeping can leave no completion
    // event even though the consumer stream is healthy.
    // Synchronize only this exceptional path before dropping its protection.
    // A failed synchronization means GPU completion cannot be established.
    if (cudaStreamSynchronize(stream) != cudaSuccess)
      std::terminate();
    completion.clear();
  }
  auto& shard = shardFor(entry->key);
  Completions previous;
  {
    Lock lock(*this, shard);
    previous.splice(previous.end(), entry->consumers);
  }
  // Our pin protects temporarily detached completion nodes. No allocations
  // are needed, and an unqueryable event stays protected for reclamation.
  for (auto it = previous.begin(); it != previous.end();) {
    if (cudaEventQuery(it->event) == cudaSuccess)
      it = previous.erase(it);
    else
      ++it;
  }
  {
    Lock lock(*this, shard);
    entry->consumers.splice(entry->consumers.end(), previous);
    entry->consumers.splice(entry->consumers.end(), completion);
    VELOX_CHECK_GT(entry->pins, 0);
    // A failed acquisition can unwind while the creator removes a failed fill.
    // That entry no longer has a queue node to return to the idle list.
    if (--entry->pins == 0 && entry->state != State::Failed) {
      shard.idle.splice(shard.idle.end(), shard.pinned, entry->position);
      eligible_ += entry->bytes;
    }
  }
}

} // namespace facebook::velox::cudf_velox
