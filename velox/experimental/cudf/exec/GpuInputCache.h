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
// Reconstructible GPU payload cache. Shares one allocator/arbitrator with
// queries.
#pragma once
#include "velox/experimental/cudf/CudaEventPool.h"
#include "velox/experimental/cudf/exec/CudfMemoryResource.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/common/memory/DiscardableMemoryReclaimer.h"
#include "velox/common/memory/Memory.h"

#include <cuda_runtime_api.h>

#include <folly/json.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace facebook::velox::memory {
class SharedArbitrator;
}

namespace facebook::velox::cudf_velox {

// Lookup and pin/evict transitions are atomic under one shard lock. An idle
// list contains only unpinned entries; no scan through pinned entries is
// needed. Each stream lease records completion before dropping its pin.
// Reclamation detaches victims, then waits/frees OUTSIDE metadata locks. No
// device-wide sync.
class GpuInputCache : public std::enable_shared_from_this<GpuInputCache> {
 public:
  class Lease;
  using LeasePtr = std::shared_ptr<Lease>;
  /// Share resource capacity with queries. Enable discardable reclamation on
  /// the resource so admission uses spare capacity and never spills operators.
  static std::shared_ptr<GpuInputCache> create(
      std::shared_ptr<memory::CustomMemoryResource> resource,
      cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
      bool profile = false);
  ~GpuInputCache() noexcept;

  /// Use a stable source generation/range key and the same byte contract on
  /// every access. The fill callback enqueues writes on stream and must retain
  /// its source until completion. Allocation refusal returns nullptr for the
  /// caller's ordinary copy path; payload failures propagate.
  LeasePtr getOrFill(
      const std::string& key,
      uint64_t bytes,
      cudaStream_t stream,
      const std::function<void(void*)>& fill,
      std::shared_ptr<const void> sourceIdentity = nullptr);
  /// Reserve a destination for asynchronous filling. Set needsFill only for
  /// the creator, which must call completeFill() or failFill() on its lease.
  /// Return nullptr for an existing unfinished fill so independent readers can
  /// bypass it without sharing failures or waiting on batches not yet
  /// submitted. Deduplicate reservations within a batch before calling this
  /// method.
  LeasePtr reserve(
      const std::string& key,
      uint64_t bytes,
      cudaStream_t stream,
      std::shared_ptr<const void> sourceIdentity,
      bool& needsFill);
  /// Pin an existing entry, or return nullptr on a miss or concurrent eviction.
  LeasePtr get(const std::string& key, cudaStream_t stream) {
    return access(key, 0, stream, nullptr);
  }
  /// Free at least target bytes when available; zero visits all idle entries.
  /// Wait for recorded consumers outside metadata locks. Return bytes freed.
  uint64_t evict(uint64_t target);
  /// Return cumulative counters and current live/eligible bytes.
  folly::dynamic snapshot() const;
  /// Return the discardable root used by the shared arbitrator.
  memory::MemoryPool* pool() const {
    return memory_->root.get();
  }
  /// Convert a CUDA failure into a Velox exception.
  static void check(cudaError_t status) {
    VELOX_CHECK_EQ(
        static_cast<int>(status), 0, "{}", cudaGetErrorString(status));
  }

 private:
  struct Entry;
  using Queue = std::list<std::shared_ptr<Entry>>;
  enum class State { Filling, Ready, Failed, Evicting };
  struct Completion {
    Completion() = default;
    Completion(const Completion&) = delete;
    Completion& operator=(const Completion&) = delete;
    PooledCudaEvent handle;
    cudaEvent_t event{handle.get()};
  };
  using Completions = std::list<Completion>;
  struct Entry {
    std::string key;
    // Keeps a source identity alive independently of host payload residency.
    std::shared_ptr<const void> sourceIdentity;
    uint64_t bytes{0};
    uint64_t pins{1};
    void* data{nullptr};
    cudaEvent_t ready{};
    State state{State::Filling};
    // Publish the destination before asynchronous host reads complete.
    bool allocated{false};
    std::condition_variable filled;
    Completions consumers;
    Queue::iterator position;
  };
  struct Shard {
    std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<Entry>> entries;
    Queue pinned;
    Queue idle;
  };
  static constexpr size_t kShards = 16;
  using Clock = std::chrono::steady_clock;
  static uint64_t now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               Clock::now().time_since_epoch())
        .count();
  }
  // Timing opt-in: metadata profiling is not silently charged to baseline runs.
  class Lock {
   public:
    Lock(GpuInputCache& cache, Shard& shard);
    ~Lock();

   private:
    GpuInputCache& cache_;
    std::unique_lock<std::mutex> lock_;
    uint64_t acquired_{0};
  };
  class Reclaimer final : public memory::DiscardableMemoryReclaimer {
   public:
    explicit Reclaimer(std::weak_ptr<GpuInputCache> cache)
        : DiscardableMemoryReclaimer(0), cache_(std::move(cache)) {}
    bool reclaimableBytes(const memory::MemoryPool&, uint64_t& bytes)
        const override;
    uint64_t reclaim(
        memory::MemoryPool*,
        uint64_t target,
        uint64_t,
        Stats& stats) override;
    void abort(memory::MemoryPool*, const std::exception_ptr&) override;

   private:
    const std::weak_ptr<GpuInputCache> cache_;
  };
  Shard& shardFor(const std::string& key) {
    return shards_[std::hash<std::string>{}(key) % kShards];
  }
  LeasePtr access(
      const std::string& key,
      uint64_t bytes,
      cudaStream_t stream,
      const std::function<void(void*)>* fill,
      std::shared_ptr<const void> sourceIdentity = nullptr,
      bool* needsFill = nullptr);
  void release(
      const std::shared_ptr<Entry>& entry,
      cudaStream_t stream,
      Completions& completion) noexcept;
  struct Memory {
    std::shared_ptr<memory::CustomMemoryResource> resource;
    std::shared_ptr<memory::MemoryPool> root;
    std::shared_ptr<memory::MemoryPool> leaf;
    std::optional<CudfMemoryResource> reporting;
  };
  explicit GpuInputCache(
      std::shared_ptr<memory::CustomMemoryResource> resource,
      bool profile);
  void initialize(cuda::mr::any_resource<cuda::mr::device_accessible> upstream);
  std::unique_ptr<Memory> memory_;
  memory::SharedArbitrator* arbitrator_;
  std::atomic<bool> initialized_{false};
  std::array<Shard, kShards> shards_;
  cudaStream_t freeStream_{};
  std::mutex freeMutex_;
  bool profile_;
  // Stop admitting new allocations after the cache pool is aborted.
  std::atomic<bool> admissionDisabled_{false};
  std::atomic<uint64_t> admitAfterNs_{0};
  std::atomic<uint64_t> admitted_{0};
  std::atomic<uint64_t> admissionFailures_{0};
  std::atomic<uint64_t> fillErrors_{0};
  std::atomic<uint64_t> cursor_{0};
  std::atomic<uint64_t> eligible_{0};
  std::atomic<uint64_t> pendingFree_{0};
  std::atomic<uint64_t> hits_{0};
  std::atomic<uint64_t> misses_{0};
  std::atomic<uint64_t> coalesced_{0};
  std::atomic<uint64_t> bypasses_{0};
  std::atomic<uint64_t> calls_{0};
  std::atomic<uint64_t> requested_{0};
  std::atomic<uint64_t> freed_{0};
  std::atomic<uint64_t> inspected_{0};
  std::atomic<uint64_t> shardsVisited_{0};
  std::atomic<uint64_t> aborts_{0};
  std::atomic<uint64_t> lockWaitNs_{0};
  std::atomic<uint64_t> lockHoldNs_{0};
  std::atomic<uint64_t> eventWaitNs_{0};
  std::atomic<uint64_t> freeNs_{0};
  std::atomic<uint64_t> reclaimNs_{0};
};

/// Pin a device allocation through work submitted on one consumer stream.
class GpuInputCache::Lease {
 public:
  Lease(std::shared_ptr<GpuInputCache> owner, cudaStream_t stream)
      : owner_(std::move(owner)), stream_(stream) {}
  Lease(Lease&&) noexcept = default;
  // Entry is private to the cache. A stack lease guards the pin immediately;
  // allocating the returned shared lease cannot abandon that ownership.
  void setEntry(std::shared_ptr<Entry> entry) noexcept {
    entry_ = std::move(entry);
  }
  void prepareCompletion();
  /// Wait for host submission of this entry's fill, propagating failure.
  void waitForFill();
  /// Publish a reservation after all writes have been enqueued on its stream.
  void completeFill();
  /// Drain writes and discard an unpublished reservation after a read
  /// failure.
  void failFill();
  ~Lease();
  Lease(const Lease&) = delete;
  Lease& operator=(const Lease&) = delete;
  /// Return the pinned device address.
  void* data() const {
    return entry_->data;
  }
  /// Return allocated bytes, including format padding.
  uint64_t size() const {
    return entry_->bytes;
  }
  /// Order the first consumer after the recorded fill. Call before use on
  /// the consumer stream; this does not join previously submitted consumers.
  void waitForFillOn(cudaStream_t stream);
  /// Transfer subsequent use to another stream in order. One lease has one
  /// CPU owner; independent consumers obtain separate leases through get().
  void handoffTo(cudaStream_t stream);

 private:
  std::shared_ptr<GpuInputCache> owner_;
  std::shared_ptr<Entry> entry_;
  cudaStream_t stream_;
  Completions completion_;
};
} // namespace facebook::velox::cudf_velox
