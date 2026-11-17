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

#include "velox/experimental/cudf/connectors/hive/BufferedInputDataSource.h"
#include "velox/experimental/cudf/exec/EncodedInputGpuCache.h"
#include "velox/experimental/cudf/exec/GpuEncodedCache.h"
#include "velox/experimental/cudf/io/CachedDeviceRead.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/file/File.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/common/memory/tests/SharedArbitratorTestUtil.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/Driver.h"
#include "velox/exec/MemoryReclaimer.h"
#include "velox/exec/Task.h"

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <folly/ScopeGuard.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/executors/InlineExecutor.h>
#include <folly/executors/ManualExecutor.h>
#include <folly/synchronization/Baton.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <future>
#include <iostream>
#include <thread>

namespace facebook::velox::cudf_velox {
namespace {
class GpuInputCacheTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    memory::MemoryManager::initialize({});
    setCudfContextDevice(0);
  }
};

TEST_F(GpuInputCacheTest, ShardedCachePinsAndStreamCompletion) {
  setCudfContextDevice(0);
  auto owner = createCudfCustomMemoryResource(128L << 20, true);
  auto upstream = createMemoryResource("cuda", 0);
  auto cache = std::make_shared<GpuEncodedCache>(owner, upstream, true);
  cudaStream_t producer, consumer;
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), cudaSuccess);
  auto lease =
      cache->getOrFill("generation:range", 8L << 20, producer, [&](void* p) {
        GpuEncodedCache::check(cudaMemsetAsync(p, 0x5a, 8L << 20, producer));
      });
  ASSERT_TRUE(lease);
  auto second = cache->get("generation:range", consumer);
  ASSERT_TRUE(second);
  EXPECT_EQ(second->data(), lease->data());
  EXPECT_EQ(cache->evict(8L << 20), 0);
  lease->handoffTo(consumer);
  unsigned char value = 0;
  GpuEncodedCache::check(cudaMemcpyAsync(
      &value, lease->data(), 1, cudaMemcpyDeviceToHost, consumer));
  lease.reset();
  EXPECT_EQ(cache->evict(8L << 20), 0);
  second.reset();
  EXPECT_EQ(cache->evict(8L << 20), 8L << 20);
  EXPECT_EQ(value, 0x5a); // Eviction waited for all consumers before freeing.
  EXPECT_EQ(cache->pool()->usedBytes(), 0);
  EXPECT_EQ(cache->snapshot()["inspected_entries"].asInt(), 1);
  cache.reset();
  cudaStreamDestroy(consumer);
  cudaStreamDestroy(producer);
}

TEST_F(GpuInputCacheTest, abortedCacheReleasesIdleAndBypassesAdmission) {
  auto resource = createCudfCustomMemoryResource(32L << 20, true);
  auto upstream = createMemoryResource("cuda", 0);
  auto cache = std::make_shared<GpuEncodedCache>(resource, upstream);
  rmm::cuda_stream stream;
  auto fill = [&](void* data) {
    GpuEncodedCache::check(
        cudaMemsetAsync(data, 0x2a, 8L << 20, stream.value()));
  };
  auto pinned = cache->getOrFill("pinned", 8L << 20, stream.value(), fill);
  auto idle = cache->getOrFill("idle", 8L << 20, stream.value(), fill);
  ASSERT_TRUE(pinned);
  ASSERT_TRUE(idle);
  idle.reset();
  cache->pool()->abort(
      std::make_exception_ptr(std::runtime_error("test abort")));
  EXPECT_EQ(cache->pool()->usedBytes(), 8L << 20);
  EXPECT_NO_THROW(EXPECT_EQ(
      cache->getOrFill("new", 8L << 20, stream.value(), fill), nullptr));
  uint8_t value = 0;
  GpuEncodedCache::check(cudaMemcpyAsync(
      &value, pinned->data(), 1, cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  EXPECT_EQ(value, 0x2a);
  pinned.reset();
  EXPECT_EQ(cache->pool()->usedBytes(), 0);
}

TEST_F(GpuInputCacheTest, admissionReplacesOnlyIdleEntries) {
  auto resource = createCudfCustomMemoryResource(16L << 20, true);
  auto upstream = createMemoryResource("cuda", 0);
  auto cache = std::make_shared<GpuEncodedCache>(resource, upstream);
  rmm::cuda_stream stream;
  auto fill = [&](void* data) {
    GpuEncodedCache::check(
        cudaMemsetAsync(data, 0x37, 8L << 20, stream.value()));
  };
  auto pinned = cache->getOrFill("pinned", 8L << 20, stream.value(), fill);
  auto idle = cache->getOrFill("idle", 8L << 20, stream.value(), fill);
  ASSERT_TRUE(pinned);
  ASSERT_TRUE(idle);
  idle.reset();
  auto replacement =
      cache->getOrFill("replacement", 8L << 20, stream.value(), fill);
  EXPECT_TRUE(replacement);
  EXPECT_EQ(cache->get("idle", stream.value()), nullptr);
  EXPECT_TRUE(cache->get("pinned", stream.value()));
  EXPECT_EQ(cache->pool()->usedBytes(), 16L << 20);
}

TEST_F(GpuInputCacheTest, ShardedCacheCoalescesConcurrentFill) {
  setCudfContextDevice(0);
  auto owner = createCudfCustomMemoryResource(128L << 20, true);
  auto upstream = createMemoryResource("cuda", 0);
  auto cache = std::make_shared<GpuEncodedCache>(owner, upstream, true);
  std::promise<void> entered, finish;
  auto gate = finish.get_future().share();
  auto creator = std::async(std::launch::async, [&] {
    ensureCudaContextForThread();
    cudaStream_t stream;
    GpuEncodedCache::check(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto lease = cache->getOrFill("same", 8L << 20, stream, [&](void* p) {
      entered.set_value();
      gate.wait();
      GpuEncodedCache::check(cudaMemsetAsync(p, 0x39, 8L << 20, stream));
    });
    auto* ptr = lease->data();
    lease.reset();
    cudaStreamDestroy(stream);
    return ptr;
  });
  entered.get_future().wait();
  auto waiter = std::async(std::launch::async, [&] {
    ensureCudaContextForThread();
    cudaStream_t stream;
    GpuEncodedCache::check(
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto lease = cache->getOrFill("same", 8L << 20, stream, [&](void*) {
      ADD_FAILURE() << "duplicate fill";
    });
    unsigned char byte = 0;
    GpuEncodedCache::check(cudaMemcpyAsync(
        &byte, lease->data(), 1, cudaMemcpyDeviceToHost, stream));
    GpuEncodedCache::check(cudaStreamSynchronize(stream));
    EXPECT_EQ(byte, 0x39);
    auto* ptr = lease->data();
    lease.reset();
    cudaStreamDestroy(stream);
    return ptr;
  });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!cache->snapshot()["coalesced_fills"].asInt() &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  EXPECT_EQ(cache->snapshot()["coalesced_fills"].asInt(), 1);
  finish.set_value();
  EXPECT_EQ(creator.get(), waiter.get());
  EXPECT_EQ(cache->snapshot()["misses"].asInt(), 1);
  EXPECT_EQ(cache->evict(0), 8L << 20);
}

enum class FillInterruption {
  kPause,
  kCancelBeforeWait,
  kCancelWhileWaiting,
};

// Exercise Task accounting with actual allocation arbitration and driver
// contexts. Keep the creator gated until the requested interruption is
// observed.
void runCoalescedFillInterruption(FillInterruption interruption) {
  for (bool failAllocation : {false, true}) {
    SCOPED_TRACE(failAllocation);
    folly::Baton<> creatorArbitrating;
    folly::Baton<> finishArbitration;
    folly::Baton<> waiterEntered;
    folly::Baton<> startLookup;
    auto prototype = createCudfCustomMemoryResource(128L << 20, true);
    auto config = prototype->arbitrator()->config();
    config.arbitrationStateCheckCb = [&](memory::MemoryPool& pool) {
      exec::memoryArbitrationStateCheck(pool);
      creatorArbitrating.post();
      finishArbitration.wait();
      if (failAllocation) {
        throw std::bad_alloc{};
      }
    };
    auto resource = std::make_shared<memory::CustomMemoryResource>(
        prototype->tag(),
        std::shared_ptr<memory::MemoryAllocator>(
            prototype, prototype->allocator()),
        std::make_shared<memory::SharedArbitrator>(config),
        [prototype](const memory::ReclaimerContext& context) {
          return prototype->newReclaimer(context);
        },
        prototype->maxCapacity());
    auto cache = std::make_shared<GpuEncodedCache>(
        resource, createMemoryResource("cuda", 0));
    auto pool = memory::memoryManager()->addLeafPool();
    auto values = std::make_shared<RowVector>(
        pool.get(), ROW({}), nullptr, 1, std::vector<VectorPtr>{});
    core::PlanFragment plan;
    plan.planNode = std::make_shared<core::ValuesNode>(
        "values", std::vector<RowVectorPtr>{values});
    folly::CPUThreadPoolExecutor executor(2);
    auto task = exec::Task::create(
        "cache-pause",
        std::move(plan),
        0,
        core::QueryCtx::create(&executor),
        exec::Task::ExecutionMode::kParallel);
    std::array<std::shared_ptr<exec::Driver>, 2> drivers;
    auto runDriver = [&](int id, auto action) {
      auto driver = exec::Driver::testingCreate(
          std::make_unique<exec::DriverCtx>(task, id, 0, 0, 0));
      drivers[id] = driver;
      driver->state().isEnqueued = true;
      VELOX_CHECK_EQ(task->enter(driver->state()), exec::StopReason::kNone);
      SCOPE_EXIT {
        task->leave(driver->state(), [](exec::StopReason) {});
      };
      exec::ScopedDriverThreadContext context(driver->driverCtx());
      ensureCudaContextForThread();
      rmm::cuda_stream stream;
      return action(stream);
    };
    auto creator = std::async(std::launch::async, [&] {
      return runDriver(0, [&](rmm::cuda_stream& stream) {
        auto lease = cache->getOrFill(
            "shared", 8L << 20, stream.value(), [&](void* destination) {
              GpuEncodedCache::check(
                  cudaMemsetAsync(destination, 0x39, 8L << 20, stream.value()));
            });
        if (failAllocation) {
          EXPECT_FALSE(lease);
          return static_cast<void*>(nullptr);
        }
        VELOX_CHECK_NOT_NULL(lease);
        unsigned char byte{0};
        GpuEncodedCache::check(cudaMemcpyAsync(
            &byte, lease->data(), 1, cudaMemcpyDeviceToHost, stream.value()));
        GpuEncodedCache::check(cudaStreamSynchronize(stream.value()));
        EXPECT_EQ(byte, 0x39);
        auto* address = lease->data();
        lease.reset();
        return address;
      });
    });
    auto waiter = std::async(std::launch::async, [&] {
      return runDriver(1, [&](rmm::cuda_stream& stream) {
        waiterEntered.post();
        startLookup.wait();
        if (interruption == FillInterruption::kCancelBeforeWait) {
          [&] {
            VELOX_ASSERT_THROW(
                cache->getOrFill(
                    "shared",
                    8L << 20,
                    stream.value(),
                    [](void*) { ADD_FAILURE() << "duplicate fill"; }),
                "Terminate detected when entering suspension");
          }();
          return static_cast<void*>(nullptr);
        }
        auto lease =
            cache->getOrFill("shared", 8L << 20, stream.value(), [](void*) {
              ADD_FAILURE() << "duplicate fill";
            });
        if (failAllocation) {
          EXPECT_FALSE(lease);
          return static_cast<void*>(nullptr);
        }
        VELOX_CHECK_NOT_NULL(lease);
        unsigned char byte{0};
        GpuEncodedCache::check(cudaMemcpyAsync(
            &byte, lease->data(), 1, cudaMemcpyDeviceToHost, stream.value()));
        GpuEncodedCache::check(cudaStreamSynchronize(stream.value()));
        EXPECT_EQ(byte, 0x39);
        auto* address = lease->data();
        lease.reset();
        return address;
      });
    });
    SCOPE_EXIT {
      finishArbitration.post();
      startLookup.post();
      exec::Task::resume(task);
    };
    ASSERT_TRUE(creatorArbitrating.try_wait_for(std::chrono::seconds(5)));
    ASSERT_TRUE(waiterEntered.try_wait_for(std::chrono::seconds(5)));
    if (interruption == FillInterruption::kPause) {
      auto paused = task->requestPause();
      finishArbitration.post();
      startLookup.post();
      EXPECT_TRUE(std::move(paused).wait(std::chrono::seconds(5)))
          << "coalesced waiter prevents task pause";
      // Resume even on failure so the broken control terminates cleanly.
      exec::Task::resume(task);
      EXPECT_EQ(creator.get(), waiter.get());
    } else {
      if (interruption == FillInterruption::kCancelWhileWaiting) {
        startLookup.post();
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!drivers[1]->state().suspended() &&
               std::chrono::steady_clock::now() < deadline) {
          std::this_thread::yield();
        }
        ASSERT_TRUE(drivers[1]->state().suspended());
        EXPECT_EQ(cache->snapshot()["coalesced_fills"].asInt(), 1);
      }
      auto canceled = task->requestCancel();
      startLookup.post();
      EXPECT_TRUE(std::move(canceled).wait(std::chrono::seconds(5)))
          << "cancellation acknowledgment depends on the pending fill";
      EXPECT_EQ(task->state(), exec::TaskState::kCanceled);
      EXPECT_EQ(cache->evict(0), 0);
      EXPECT_EQ(cache->snapshot()["eligible_bytes"].asInt(), 0);
      if (interruption == FillInterruption::kCancelBeforeWait) {
        // Rejected suspension must unwind the waiter before the creator runs.
        const auto status = waiter.wait_for(std::chrono::seconds(5));
        EXPECT_EQ(status, std::future_status::ready);
        if (status != std::future_status::ready) {
          // Unblock cleanup even if the cancellation contract regresses.
          finishArbitration.post();
        }
        EXPECT_EQ(waiter.get(), nullptr);
        EXPECT_EQ(drivers[1]->state().numSuspensions.load(), 0);
        EXPECT_FALSE(drivers[1]->state().isOnThread());
        finishArbitration.post();
        EXPECT_EQ(creator.get() == nullptr, failAllocation);
      } else {
        // Cancellation acknowledges a suspended driver. Its stack and pin
        // remain alive until the producer completes or refuses allocation.
        EXPECT_EQ(
            waiter.wait_for(std::chrono::seconds(0)),
            std::future_status::timeout);
        finishArbitration.post();
        EXPECT_EQ(creator.get(), waiter.get());
      }
      for (const auto& driver : drivers) {
        EXPECT_EQ(driver->state().numSuspensions.load(), 0);
        EXPECT_FALSE(driver->state().isOnThread());
        EXPECT_TRUE(driver->state().isTerminated);
      }
    }
    EXPECT_EQ(cache->snapshot()["coalesced_fills"].asInt(), 1);
    EXPECT_EQ(cache->evict(0), failAllocation ? 0 : 8L << 20);
    EXPECT_EQ(cache->pool()->usedBytes(), 0);
    EXPECT_EQ(cache->snapshot()["eligible_bytes"].asInt(), 0);
    EXPECT_EQ(cache->snapshot()["pending_free_bytes"].asInt(), 0);
  }
}

TEST_F(GpuInputCacheTest, coalescedFillAllowsTaskPause) {
  runCoalescedFillInterruption(FillInterruption::kPause);
}

TEST_F(GpuInputCacheTest, coalescedFillCancelsBeforeSuspension) {
  runCoalescedFillInterruption(FillInterruption::kCancelBeforeWait);
}

TEST_F(GpuInputCacheTest, coalescedFillCancelsWhileWaiting) {
  runCoalescedFillInterruption(FillInterruption::kCancelWhileWaiting);
}

TEST_F(GpuInputCacheTest, ShardedAdmissionBypassesWithoutSpilling) {
  setCudfContextDevice(0);
  auto owner = createCudfCustomMemoryResource(64L << 20, true);
  auto upstream = createMemoryResource("cuda", 0);
  auto root = memory::memoryManager()->addCustomRootPool("active-query", owner);
  auto leaf = root->addLeafChild("state");
  CudfMemoryResource query(upstream, leaf);
  auto* data = query.allocate_sync(56L << 20, alignof(std::max_align_t));
  auto cache = std::make_shared<GpuEncodedCache>(owner, upstream);
  auto lease = cache->getOrFill(
      "cannot-fit", 16L << 20, nullptr, [](void*) { ADD_FAILURE(); });
  EXPECT_FALSE(lease);
  EXPECT_EQ(cache->pool()->usedBytes(), 0);
  EXPECT_EQ(cache->snapshot()["bypasses"].asInt(), 1);
  EXPECT_EQ(cache->snapshot()["cache_aborts"].asInt(), 0);
  EXPECT_EQ(leaf->usedBytes(), 56L << 20);
  query.deallocate_sync(data, 56L << 20, alignof(std::max_align_t));
}

TEST_F(GpuInputCacheTest, pinnedAdmissionAtParticipantLimitBypasses) {
  setCudfContextDevice(0);
  auto owner = createCudfCustomMemoryResource(64L << 20, true);
  auto cache =
      std::make_shared<GpuEncodedCache>(owner, createMemoryResource("cuda", 0));
  auto first = cache->getOrFill("retained", 56L << 20, nullptr, [](void* p) {
    GpuEncodedCache::check(cudaMemsetAsync(p, 0x5a, 56L << 20));
  });
  ASSERT_TRUE(first);
  // Keep the old entry pinned: replacement may reclaim only idle entries.
  GpuEncodedCache::LeasePtr next;
  EXPECT_NO_THROW(
      next = cache->getOrFill(
          "over-participant-limit", 16L << 20, nullptr, [](void*) {
            ADD_FAILURE();
          }));
  EXPECT_FALSE(next);
  EXPECT_EQ(cache->pool()->usedBytes(), 56L << 20);
  EXPECT_EQ(cache->snapshot()["bypasses"].asInt(), 1);
  EXPECT_EQ(cache->snapshot()["freed_bytes"].asInt(), 0);
  EXPECT_EQ(cache->snapshot()["cache_aborts"].asInt(), 0);
  EXPECT_EQ(cache->snapshot()["fill_errors"].asInt(), 0);
  auto retained = cache->get("retained", nullptr);
  ASSERT_TRUE(retained);
  unsigned char value = 0;
  GpuEncodedCache::check(
      cudaMemcpy(&value, retained->data(), 1, cudaMemcpyDeviceToHost));
  EXPECT_EQ(value, 0x5a);
  retained.reset();
  first.reset();
  EXPECT_EQ(cache->evict(0), 56L << 20);
  EXPECT_EQ(cache->pool()->usedBytes(), 0);
}

TEST_F(GpuInputCacheTest, ShardedFillFailureCanRetry) {
  setCudfContextDevice(0);
  auto owner = createCudfCustomMemoryResource(128L << 20, true);
  auto upstream = createMemoryResource("cuda", 0);
  auto cache = std::make_shared<GpuEncodedCache>(owner, upstream);
  EXPECT_THROW(
      cache->getOrFill(
          "retry",
          8L << 20,
          nullptr,
          [](void*) { throw std::runtime_error("fill failure"); }),
      std::runtime_error);
  EXPECT_EQ(cache->pool()->usedBytes(), 0);
  auto lease = cache->getOrFill("retry", 8L << 20, nullptr, [](void* p) {
    GpuEncodedCache::check(cudaMemsetAsync(p, 7, 8L << 20));
  });
  EXPECT_TRUE(lease);
  lease.reset();
  EXPECT_EQ(cache->evict(0), 8L << 20);
}

// Advertise the host cache capability in adapter unit tests. The all-query
// integration test exercises the actual OSS CachedBufferedInput implementation.
class CacheableInput : public dwio::common::BufferedInput {
 public:
  CacheableInput(
      std::shared_ptr<ReadFile> file,
      memory::MemoryPool& pool,
      bool cacheable,
      folly::Executor* executor = &folly::InlineExecutor::instance())
      : BufferedInput(std::move(file), pool),
        cacheable_(cacheable),
        executor_(executor) {}
  folly::Executor* executor() const override {
    return executor_;
  }
  bool hasCache() const override {
    return cacheable_;
  }

 private:
  bool cacheable_;
  folly::Executor* executor_;
};

// Records physical host requests while exercising the normal datasource API.
class RecordingInputSource : public connector::hive::BufferedInputDataSource {
 public:
  using BufferedInputDataSource::BufferedInputDataSource;
  std::vector<std::pair<size_t, size_t>> reads;
  std::optional<size_t> failOffset;
  std::mutex readsMutex;
  std::function<void()> beforeRead;
  std::unique_ptr<buffer> suppliedBuffer;
  std::unique_ptr<buffer> host_read(size_t offset, size_t bytes) override {
    {
      std::lock_guard lock(readsMutex);
      reads.emplace_back(offset, bytes);
    }
    if (beforeRead) {
      beforeRead();
    }
    if (failOffset == offset) {
      throw std::runtime_error("injected cached read failure");
    }
    if (suppliedBuffer) {
      return std::move(suppliedBuffer);
    }
    return BufferedInputDataSource::host_read(offset, bytes);
  }
};

TEST_F(GpuInputCacheTest, coldReadsReturnPendingAndCoalesceMisses) {
  auto resource = createCudfCustomMemoryResource(128L << 20, true);
  auto upstream = createMemoryResource("cuda", 0);
  auto cache = std::make_shared<GpuEncodedCache>(resource, upstream);
  setEncodedInputGpuCache(cache);
  SCOPE_EXIT {
    setEncodedInputGpuCache(nullptr);
  };
  auto pool = memory::memoryManager()->addLeafPool();
  StringIdMap ids;
  auto identity = std::make_shared<StringIdLease>(ids, "cold-input");
  folly::ManualExecutor executor;
  auto input = std::make_shared<CacheableInput>(
      std::make_shared<InMemoryReadFile>(std::string(32, 'x')),
      *pool,
      true,
      &executor);
  auto source =
      std::make_shared<RecordingInputSource>(input, nullptr, identity);
  rmm::cuda_stream stream;
  rmm::device_async_resource_ref mr{upstream};
  const std::array<gpu::DeviceByteRange, 3> ranges{{{0, 4}, {4, 4}, {16, 4}}};
  auto batch = source->readDeviceRangesAsync(ranges, false, stream, mr, mr);
  EXPECT_TRUE(batch.pending.valid());
  EXPECT_TRUE(source->reads.empty());
  batch.waitOn(stream);
  EXPECT_EQ(source->reads.size(), 2);
  if (source->reads.size() == 2) {
    EXPECT_EQ(source->reads[0], (std::pair<size_t, size_t>{0, 8}));
    EXPECT_EQ(source->reads[1], (std::pair<size_t, size_t>{16, 4}));
  }
  for (const auto& view : batch.data) {
    uint8_t byte = 0;
    GpuEncodedCache::check(cudaMemcpyAsync(
        &byte, view.data(), 1, cudaMemcpyDeviceToHost, stream.value()));
    stream.synchronize();
    EXPECT_EQ(byte, 'x');
  }
  executor.drain();
}

void testUploadRetirement(bool cacheEnabled) {
  class PinnedBuffer final : public cudf::io::datasource::buffer {
   public:
    explicit PinnedBuffer(std::promise<void>& freed) : freed_(freed) {
      GpuEncodedCache::check(cudaMallocHost(&data_, 8));
      std::memset(data_, 'x', 8);
    }
    ~PinnedBuffer() override {
      cudaFreeHost(data_);
      freed_.set_value();
    }
    const uint8_t* data() const override {
      return data_;
    }
    size_t size() const override {
      return 8;
    }

   private:
    uint8_t* data_;
    std::promise<void>& freed_;
  };
  auto resource = createCudfCustomMemoryResource(128L << 20, true);
  auto upstream = createMemoryResource("cuda", 0);
  auto cache = std::make_shared<GpuEncodedCache>(resource, upstream);
  setEncodedInputGpuCache(cacheEnabled ? cache : nullptr);
  SCOPE_EXIT {
    setEncodedInputGpuCache(nullptr);
  };
  auto pool = memory::memoryManager()->addLeafPool();
  StringIdMap ids;
  folly::CPUThreadPoolExecutor executor(1);
  auto input = std::make_shared<CacheableInput>(
      std::make_shared<InMemoryReadFile>(std::string(8, 'x')),
      *pool,
      true,
      &executor);
  auto source = std::make_shared<RecordingInputSource>(
      input, nullptr, std::make_shared<StringIdLease>(ids, "pinned-async"));
  std::promise<void> freed;
  auto freedFuture = freed.get_future();
  source->suppliedBuffer = std::make_unique<PinnedBuffer>(freed);
  rmm::cuda_stream stream;
  rmm::device_async_resource_ref mr{upstream};
  // Reserve first so allocation is not hidden behind the blocked stream.
  const std::array<gpu::DeviceByteRange, 1> ranges{{{0, 8}}};
  std::promise<void> unblock;
  auto gate = unblock.get_future();
  source->beforeRead = [&] {
    GpuEncodedCache::check(cudaLaunchHostFunc(
        stream.value(),
        [](void* value) { static_cast<std::future<void>*>(value)->wait(); },
        &gate));
  };
  auto batch = source->readDeviceRangesAsync(ranges, false, stream, mr, mr);
  auto completion = std::async(std::launch::async, [&] { batch.wait(); });
  const auto status = completion.wait_for(std::chrono::seconds(2));
  EXPECT_EQ(
      freedFuture.wait_for(std::chrono::seconds(0)),
      std::future_status::timeout);
  // A blocked GPU must not occupy the only thread serving physical reads.
  std::promise<void> ioProgress;
  auto ioFuture = ioProgress.get_future();
  executor.add([&] { ioProgress.set_value(); });
  const auto ioStatus = ioFuture.wait_for(std::chrono::seconds(2));
  unblock.set_value();
  completion.get();
  EXPECT_EQ(status, std::future_status::ready);
  executor.join();
  EXPECT_EQ(ioStatus, std::future_status::ready);
  EXPECT_EQ(
      freedFuture.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  freedFuture.get();
  uint8_t value = 0;
  GpuEncodedCache::check(cudaMemcpyAsync(
      &value, batch.data[0].data(), 1, cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  EXPECT_EQ(value, 'x');
}

TEST_F(GpuInputCacheTest, pendingCompletionDoesNotWaitForDevice) {
  testUploadRetirement(true);
}

TEST_F(GpuInputCacheTest, uncachedUploadsDoNotBlockIoExecutor) {
  testUploadRetirement(false);
}

TEST_F(GpuInputCacheTest, coalescedMissesHaveBoundedHostReads) {
  auto resource = createCudfCustomMemoryResource(128L << 20, true);
  auto upstream = createMemoryResource("cuda", 0);
  auto cache = std::make_shared<GpuEncodedCache>(resource, upstream);
  setEncodedInputGpuCache(cache);
  SCOPE_EXIT {
    setEncodedInputGpuCache(nullptr);
  };
  auto pool = memory::memoryManager()->addLeafPool();
  StringIdMap ids;
  folly::ManualExecutor executor;
  constexpr size_t kChunk = 8L << 20;
  auto input = std::make_shared<CacheableInput>(
      std::make_shared<InMemoryReadFile>(std::string(3 * kChunk, 'x')),
      *pool,
      true,
      &executor);
  auto source = std::make_shared<RecordingInputSource>(
      input, nullptr, std::make_shared<StringIdLease>(ids, "bounded-reads"));
  rmm::cuda_stream stream;
  rmm::device_async_resource_ref mr{upstream};
  const std::array<gpu::DeviceByteRange, 3> ranges{
      {{0, kChunk}, {kChunk, kChunk}, {2 * kChunk, kChunk}}};
  auto batch = source->readDeviceRangesAsync(ranges, false, stream, mr, mr);
  batch.waitOn(stream);
  EXPECT_EQ(source->reads.size(), 2);
  if (source->reads.size() == 2) {
    EXPECT_EQ(source->reads[0], (std::pair<size_t, size_t>{0, 2 * kChunk}));
    EXPECT_EQ(
        source->reads[1], (std::pair<size_t, size_t>{2 * kChunk, kChunk}));
  }
  executor.drain();
}

TEST_F(GpuInputCacheTest, asynchronousFillFailureDrainsAndRetries) {
  auto resource = createCudfCustomMemoryResource(128L << 20, true);
  auto upstream = createMemoryResource("cuda", 0);
  auto cache = std::make_shared<GpuEncodedCache>(resource, upstream);
  setEncodedInputGpuCache(cache);
  SCOPE_EXIT {
    setEncodedInputGpuCache(nullptr);
  };
  auto pool = memory::memoryManager()->addLeafPool();
  StringIdMap ids;
  auto identity = std::make_shared<StringIdLease>(ids, "failing-input");
  folly::ManualExecutor executor;
  auto input = std::make_shared<CacheableInput>(
      std::make_shared<InMemoryReadFile>(std::string(32, 'x')),
      *pool,
      true,
      &executor);
  auto source =
      std::make_shared<RecordingInputSource>(input, nullptr, identity);
  source->failOffset = 0;
  rmm::cuda_stream stream;
  rmm::device_async_resource_ref mr{upstream};
  const std::array<gpu::DeviceByteRange, 3> ranges{{{0, 4}, {0, 4}, {16, 4}}};
  auto batch = source->readDeviceRangesAsync(ranges, false, stream, mr, mr);
  EXPECT_THROW(batch.wait(), std::runtime_error);
  EXPECT_EQ(source->reads.size(), 2);
  batch = {};
  // The failed group is discarded; the independently successful suffix stays.
  EXPECT_EQ(cache->pool()->usedBytes(), 8);
  executor.drain();
  source->failOffset.reset();
  {
    auto retry = source->readDeviceRangesAsync(ranges, false, stream, mr, mr);
    retry.waitOn(stream);
    EXPECT_EQ(retry.data[0].data(), retry.data[1].data());
  }
  executor.drain();
  EXPECT_EQ(cache->evict(0), 16);
}

TEST_F(GpuInputCacheTest, abandonedAsyncFillCompletesBeforeRelease) {
  auto resource = createCudfCustomMemoryResource(128L << 20, true);
  auto upstream = createMemoryResource("cuda", 0);
  auto cache = std::make_shared<GpuEncodedCache>(resource, upstream);
  setEncodedInputGpuCache(cache);
  SCOPE_EXIT {
    setEncodedInputGpuCache(nullptr);
  };
  auto pool = memory::memoryManager()->addLeafPool();
  StringIdMap ids;
  folly::ManualExecutor executor;
  auto input = std::make_shared<CacheableInput>(
      std::make_shared<InMemoryReadFile>(std::string(32, 'x')),
      *pool,
      true,
      &executor);
  auto source = std::make_shared<RecordingInputSource>(
      input, nullptr, std::make_shared<StringIdLease>(ids, "abandoned-input"));
  rmm::cuda_stream stream;
  rmm::device_async_resource_ref mr{upstream};
  const std::array<gpu::DeviceByteRange, 2> ranges{{{0, 4}, {16, 4}}};
  {
    auto batch = source->readDeviceRangesAsync(ranges, false, stream, mr, mr);
    EXPECT_TRUE(source->reads.empty());
  }
  EXPECT_EQ(source->reads.size(), 2);
  executor.drain();
  EXPECT_EQ(cache->evict(0), 16);
}

TEST_F(GpuInputCacheTest, rejectedSubmissionDrainsAcceptedReads) {
  class RejectSecondSubmission : public folly::Executor {
   public:
    void add(folly::Func task) override {
      if (submissions_++ == 1) {
        throw std::runtime_error("injected submission failure");
      }
      executor.add(std::move(task));
    }
    folly::ManualExecutor executor;

   private:
    size_t submissions_{0};
  } executor;
  auto resource = createCudfCustomMemoryResource(128L << 20, true);
  auto upstream = createMemoryResource("cuda", 0);
  auto cache = std::make_shared<GpuEncodedCache>(resource, upstream);
  setEncodedInputGpuCache(cache);
  SCOPE_EXIT {
    setEncodedInputGpuCache(nullptr);
  };
  auto pool = memory::memoryManager()->addLeafPool();
  StringIdMap ids;
  auto input = std::make_shared<CacheableInput>(
      std::make_shared<InMemoryReadFile>(std::string(32, 'x')),
      *pool,
      true,
      &executor);
  auto source = std::make_shared<RecordingInputSource>(
      input, nullptr, std::make_shared<StringIdLease>(ids, "rejected-input"));
  rmm::cuda_stream stream;
  rmm::device_async_resource_ref mr{upstream};
  const std::array<gpu::DeviceByteRange, 2> ranges{{{0, 4}, {16, 4}}};
  EXPECT_THROW(
      source->readDeviceRangesAsync(ranges, false, stream, mr, mr),
      std::runtime_error);
  // The accepted prefix is complete; the unsubmitted suffix is discarded.
  EXPECT_EQ(source->reads.size(), 1);
  EXPECT_EQ(cache->pool()->usedBytes(), 8);
  executor.executor.drain();
  {
    auto retry = source->readDeviceRangesAsync(ranges, false, stream, mr, mr);
    retry.waitOn(stream);
    for (const auto& view : retry.data) {
      uint8_t byte = 0;
      GpuEncodedCache::check(cudaMemcpyAsync(
          &byte, view.data(), 1, cudaMemcpyDeviceToHost, stream.value()));
      stream.synchronize();
      EXPECT_EQ(byte, 'x');
    }
  }
  executor.executor.drain();
  EXPECT_EQ(cache->evict(0), 16);
}

TEST_F(GpuInputCacheTest, concurrentAsyncFillBypassesUnfinishedProducer) {
  auto resource = createCudfCustomMemoryResource(128L << 20, true);
  auto upstream = createMemoryResource("cuda", 0);
  auto cache = std::make_shared<GpuEncodedCache>(resource, upstream);
  setEncodedInputGpuCache(cache);
  SCOPE_EXIT {
    setEncodedInputGpuCache(nullptr);
  };
  auto pool = memory::memoryManager()->addLeafPool();
  StringIdMap ids;
  folly::CPUThreadPoolExecutor executor(2);
  auto input = std::make_shared<CacheableInput>(
      std::make_shared<InMemoryReadFile>(std::string(32, 'x')),
      *pool,
      true,
      &executor);
  auto source = std::make_shared<RecordingInputSource>(
      input,
      nullptr,
      std::make_shared<StringIdLease>(ids, "shared-async-input"));
  std::promise<void> release;
  auto gate = release.get_future().share();
  source->beforeRead = [gate] { gate.wait(); };
  rmm::cuda_stream stream;
  rmm::device_async_resource_ref mr{upstream};
  const std::array<gpu::DeviceByteRange, 1> ranges{{{0, 4}}};
  auto first = source->readDeviceRangesAsync(ranges, false, stream, mr, mr);
  auto second = source->readDeviceRangesAsync(ranges, false, stream, mr, mr);
  release.set_value();
  auto waiter = std::async(std::launch::async, [&] { second.wait(); });
  EXPECT_EQ(
      waiter.wait_for(std::chrono::seconds(5)), std::future_status::ready);
  first.wait();
  waiter.get();
  EXPECT_NE(first.data[0].data(), second.data[0].data());
  EXPECT_EQ(source->reads.size(), 2);
}

TEST_F(GpuInputCacheTest, failedFillDoesNotFailAnotherReader) {
  auto resource = createCudfCustomMemoryResource(128L << 20, true);
  auto upstream = createMemoryResource("cuda", 0);
  auto cache = std::make_shared<GpuEncodedCache>(resource, upstream);
  setEncodedInputGpuCache(cache);
  SCOPE_EXIT {
    setEncodedInputGpuCache(nullptr);
  };
  auto pool = memory::memoryManager()->addLeafPool();
  StringIdMap ids;
  folly::ManualExecutor executor;
  auto input = std::make_shared<CacheableInput>(
      std::make_shared<InMemoryReadFile>(std::string(32, 'x')),
      *pool,
      true,
      &executor);
  auto identity = std::make_shared<StringIdLease>(ids, "isolated-failure");
  auto producer =
      std::make_shared<RecordingInputSource>(input, nullptr, identity);
  auto reader =
      std::make_shared<RecordingInputSource>(input, nullptr, identity);
  producer->failOffset = 0;
  rmm::cuda_stream firstStream, secondStream;
  rmm::device_async_resource_ref mr{upstream};
  const std::array<gpu::DeviceByteRange, 1> ranges{{{0, 4}}};
  auto first =
      producer->readDeviceRangesAsync(ranges, false, firstStream, mr, mr);
  auto second =
      reader->readDeviceRangesAsync(ranges, false, secondStream, mr, mr);
  EXPECT_THROW(first.wait(), std::runtime_error);
  ASSERT_NO_THROW(second.waitOn(secondStream));
  uint8_t byte{0};
  GpuEncodedCache::check(cudaMemcpyAsync(
      &byte,
      second.data[0].data(),
      1,
      cudaMemcpyDeviceToHost,
      secondStream.value()));
  secondStream.synchronize();
  EXPECT_EQ(byte, 'x');
  executor.drain();
}

TEST_F(GpuInputCacheTest, physicalRecoveryDoesNotCountAsGlobalArbitration) {
  auto resource = createCudfCustomMemoryResource(128L << 20, true);
  auto upstream = createMemoryResource("cuda", 0);
  auto cache = std::make_shared<GpuEncodedCache>(resource, upstream);
  rmm::cuda_stream stream;
  cache
      ->getOrFill(
          "idle",
          8L << 20,
          stream.value(),
          [&](void* data) {
            GpuEncodedCache::check(
                cudaMemsetAsync(data, 0, 8L << 20, stream.value()));
          })
      .reset();
  auto* arbitrator =
      static_cast<memory::SharedArbitrator*>(resource->arbitrator());
  memory::test::SharedArbitratorTestHelper helper(arbitrator);
  const auto before = helper.globalArbitrationRuns();
  EXPECT_GE(arbitrator->reclaimDiscardableMemory(8L << 20), 8L << 20);
  EXPECT_EQ(helper.globalArbitrationRuns(), before);
  const auto recovery = arbitrator->discardableRecoveryStats();
  EXPECT_EQ(recovery.requests, 1);
  EXPECT_GE(recovery.capacityBytes, 8L << 20);
}

TEST_F(GpuInputCacheTest, cachedReadKeepsPaddingOutsideLogicalView) {
  auto resource = createCudfCustomMemoryResource(128L << 20, true);
  auto cache = std::make_shared<GpuEncodedCache>(
      resource, createMemoryResource("cuda", 0));
  rmm::cuda_stream stream;
  int fills{0};
  auto fill = [&](void* destination) {
    ++fills;
    GpuEncodedCache::check(
        cudaMemsetAsync(destination, 0x39, 5, stream.value()));
  };
  auto first = readCachedDeviceRange(cache, "padded", 5, 3, stream, fill);
  auto hit = readCachedDeviceRange(cache, "padded", 5, 3, stream, fill);
  auto supplied = readCachedDeviceRange(
      cache,
      "padded",
      5,
      3,
      stream,
      fill,
      nullptr,
      cache->get("padded", stream.value()));
  for (auto* result : {&first, &hit, &supplied}) {
    ASSERT_TRUE(result->batch);
    result->batch.waitOn(stream);
    ASSERT_EQ(result->batch.data.size(), 1);
    EXPECT_EQ(result->batch.data.front().size(), 5);
    EXPECT_EQ(
        result->batch.data.front().data(), first.batch.data.front().data());
  }
  EXPECT_EQ(first.outcome, DeviceCacheOutcome::Fill);
  EXPECT_EQ(hit.outcome, DeviceCacheOutcome::Hit);
  EXPECT_EQ(supplied.outcome, DeviceCacheOutcome::Hit);
  EXPECT_EQ(fills, 1);
  std::array<unsigned char, 8> bytes{};
  GpuEncodedCache::check(cudaMemcpyAsync(
      bytes.data(),
      first.batch.data.front().data(),
      bytes.size(),
      cudaMemcpyDeviceToHost,
      stream.value()));
  first = {};
  hit = {};
  supplied = {};
  EXPECT_EQ(cache->evict(0), 8);
  for (size_t i = 0; i < bytes.size(); ++i) {
    EXPECT_EQ(bytes[i], i < 5 ? 0x39 : 0);
  }
  EXPECT_EQ(cache->pool()->usedBytes(), 0);
}

TEST_F(GpuInputCacheTest, cachedReadRejectsMismatchedSuppliedLease) {
  auto resource = createCudfCustomMemoryResource(128L << 20, true);
  auto cache = std::make_shared<GpuEncodedCache>(
      resource, createMemoryResource("cuda", 0));
  rmm::cuda_stream stream;
  auto lease =
      cache->getOrFill("padded", 8, stream.value(), [&](void* destination) {
        GpuEncodedCache::check(
            cudaMemsetAsync(destination, 0, 8, stream.value()));
      });
  VELOX_ASSERT_THROW(
      readCachedDeviceRange(
          cache, "padded", 5, 2, stream, [](void*) {}, nullptr, lease),
      "Cached allocation size does not match requested bytes and padding");
  lease.reset();
  EXPECT_EQ(cache->evict(0), 8);
}

TEST_F(GpuInputCacheTest, retainedParquetInputIsZeroCopyAndPadded) {
  const auto resource = createCudfCustomMemoryResource(128L << 20, true);
  auto upstream = createMemoryResource("cuda", 0);
  auto cache = std::make_shared<GpuEncodedCache>(resource, upstream);
  setEncodedInputGpuCache(cache);
  SCOPE_EXIT {
    setEncodedInputGpuCache(nullptr);
  };
  auto pool = memory::memoryManager()->addLeafPool();
  StringIdMap ids;
  auto identity = std::make_shared<StringIdLease>(ids, "immutable-file");
  std::weak_ptr<const StringIdLease> weakIdentity = identity;
  auto stats = std::make_shared<IoStats>();
  auto input = std::make_shared<CacheableInput>(
      std::make_shared<InMemoryReadFile>(std::string(32, 'a')), *pool, true);
  auto source = std::make_shared<connector::hive::BufferedInputDataSource>(
      input, stats, identity);
  rmm::cuda_stream stream;
  rmm::device_async_resource_ref mr{upstream};
  const std::array<gpu::DeviceByteRange, 1> ranges{{{1, 7}}};
  auto first = source->readDeviceRangesAsync(ranges, false, stream, mr, mr);
  first.wait();
  auto second = source->readDeviceRangesAsync(ranges, false, stream, mr, mr);
  second.wait();
  ASSERT_EQ(first.data.size(), 1);
  EXPECT_EQ(first.data.front().size(), 7);
  EXPECT_EQ(first.data.front().data(), second.data.front().data());
  EXPECT_EQ(stats->stats().at("cudfInputH2DBytes").sum, 7);
  EXPECT_EQ(stats->stats().at("cudfGpuCacheFillBytes").sum, 7);
  EXPECT_EQ(stats->stats().at("cudfGpuCacheHitBytes").sum, 7);
  EXPECT_EQ(cache->evict(0), 0);
  std::array<unsigned char, 8> bytes{};
  GpuEncodedCache::check(cudaMemcpyAsync(
      bytes.data(),
      second.data.front().data(),
      bytes.size(),
      cudaMemcpyDeviceToHost,
      stream.value()));
  first = {};
  second = {};
  source.reset();
  identity.reset();
  EXPECT_FALSE(weakIdentity.expired());
  EXPECT_EQ(cache->evict(0), 8);
  EXPECT_TRUE(weakIdentity.expired());
  for (size_t i = 0; i < 7; ++i) {
    EXPECT_EQ(bytes[i], 'a');
  }
  EXPECT_EQ(bytes[7], 0);
}

TEST_F(GpuInputCacheTest, uncachedHostSourceRetainsCopyPath) {
  const auto resource = createCudfCustomMemoryResource(128L << 20, true);
  auto upstream = createMemoryResource("cuda", 0);
  auto cache = std::make_shared<GpuEncodedCache>(resource, upstream);
  setEncodedInputGpuCache(cache);
  SCOPE_EXIT {
    setEncodedInputGpuCache(nullptr);
  };
  auto pool = memory::memoryManager()->addLeafPool();
  StringIdMap ids;
  auto identity = std::make_shared<StringIdLease>(ids, "uncached-file");
  auto stats = std::make_shared<IoStats>();
  auto input = std::make_shared<CacheableInput>(
      std::make_shared<InMemoryReadFile>(std::string(32, 'a')), *pool, false);
  auto source = std::make_shared<connector::hive::BufferedInputDataSource>(
      input, stats, identity);
  rmm::cuda_stream stream;
  rmm::device_async_resource_ref mr{upstream};
  const std::array<gpu::DeviceByteRange, 1> ranges{{{0, 8}}};
  auto first = source->readDeviceRangesAsync(ranges, false, stream, mr, mr);
  first.wait();
  auto second = source->readDeviceRangesAsync(ranges, false, stream, mr, mr);
  second.wait();
  EXPECT_NE(first.data.front().data(), second.data.front().data());
  EXPECT_EQ(stats->stats().at("cudfInputH2DBytes").sum, 16);
  EXPECT_EQ(cache->pool()->usedBytes(), 0);
}

TEST_F(GpuInputCacheTest, concurrentEvictionReleasesEachAllocationOnce) {
  auto resource = createCudfCustomMemoryResource(256L << 20, true);
  auto cache = std::make_shared<GpuEncodedCache>(
      resource, createMemoryResource("cuda", 0));
  for (int i = 0; i < 64; ++i) {
    ASSERT_TRUE(
        cache->getOrFill(std::to_string(i), 1L << 20, nullptr, [](void* p) {
          GpuEncodedCache::check(cudaMemsetAsync(p, 0x39, 1L << 20));
        }));
  }
  std::vector<std::future<uint64_t>> workers;
  for (int i = 0; i < 10; ++i) {
    workers.push_back(
        std::async(std::launch::async, [&] { return cache->evict(0); }));
  }
  uint64_t freed = 0;
  for (auto& worker : workers) {
    freed += worker.get();
  }
  EXPECT_EQ(freed, 64L << 20);
  EXPECT_EQ(cache->pool()->usedBytes(), 0);
  EXPECT_EQ(cache->snapshot()["inspected_entries"].asInt(), 64);
}

TEST_F(GpuInputCacheTest, physicalAllocatorFailureRecoversCacheOnly) {
  for (bool asynchronous : {false, true}) {
    SCOPED_TRACE(asynchronous);
    auto resource = createCudfCustomMemoryResource(256L << 20, true);
    cuda::mr::any_resource<cuda::mr::device_accessible> upstream{
        rmm::mr::pool_memory_resource{
            createMemoryResource("cuda", 0), 64L << 20, 64L << 20}};
    auto cache = std::make_shared<GpuEncodedCache>(resource, upstream);
    GpuEncodedCache::LeasePtr pin;
    for (int i = 0; i < 4; ++i) {
      auto lease =
          cache->getOrFill(std::to_string(i), 12L << 20, nullptr, [](void* p) {
            GpuEncodedCache::check(cudaMemsetAsync(p, 0x5a, 12L << 20));
          });
      ASSERT_TRUE(lease);
      if (i == 0) {
        pin = std::move(lease);
      }
    }
    auto root =
        memory::memoryManager()->addCustomRootPool("physical-demand", resource);
    auto leaf = root->addLeafChild("request");
    CudfMemoryResource allocation(upstream, leaf);
    void* data = nullptr;
    ASSERT_NO_THROW(
        data = asynchronous
            ? allocation.allocate(
                  cuda::stream_ref{cudaStream_t{nullptr}},
                  32L << 20,
                  alignof(std::max_align_t))
            : allocation.allocate_sync(32L << 20, alignof(std::max_align_t)));
    ASSERT_NE(data, nullptr);
    EXPECT_GE(cache->snapshot()["freed_bytes"].asInt(), 16L << 20);
    EXPECT_EQ(resource->arbitrator()->stats().numAborted, 0);
    unsigned char byte = 0;
    GpuEncodedCache::check(
        cudaMemcpy(&byte, pin->data(), 1, cudaMemcpyDeviceToHost));
    EXPECT_EQ(byte, 0x5a);
    if (asynchronous) {
      allocation.deallocate(
          cuda::stream_ref{cudaStream_t{nullptr}},
          data,
          32L << 20,
          alignof(std::max_align_t));
    } else {
      allocation.deallocate_sync(data, 32L << 20, alignof(std::max_align_t));
    }
    pin.reset();
    cache->evict(0);
    EXPECT_EQ(cache->pool()->usedBytes(), 0);
    EXPECT_EQ(leaf->usedBytes(), 0);
  }
}

TEST_F(GpuInputCacheTest, logicalPressureUsesDiscardableStorage) {
  auto resource = createCudfCustomMemoryResource(64L << 20, true);
  auto upstream = createMemoryResource("cuda", 0);
  auto cache = std::make_shared<GpuEncodedCache>(resource, upstream);
  for (int i = 0; i < 6; ++i) {
    ASSERT_TRUE(
        cache->getOrFill(std::to_string(i), 8L << 20, nullptr, [](void* p) {
          GpuEncodedCache::check(cudaMemsetAsync(p, 0x5a, 8L << 20));
        }));
  }
  auto root =
      memory::memoryManager()->addCustomRootPool("logical-demand", resource);
  auto leaf = root->addLeafChild("request");
  CudfMemoryResource allocation(upstream, leaf);
  auto* data = allocation.allocate_sync(32L << 20, alignof(std::max_align_t));
  EXPECT_GE(cache->snapshot()["freed_bytes"].asInt(), 16L << 20);
  EXPECT_EQ(resource->arbitrator()->stats().numAborted, 0);
  EXPECT_EQ(cache->snapshot()["cache_aborts"].asInt(), 0);
  allocation.deallocate_sync(data, 32L << 20, alignof(std::max_align_t));
  cache->evict(0);
  EXPECT_EQ(cache->pool()->usedBytes(), 0);
}

TEST_F(GpuInputCacheTest, fullDeviceCapacityIsShared) {
  if (std::getenv("VELOX_GPU_CACHE_FULL_CAPACITY_TEST") == nullptr) {
    GTEST_SKIP()
        << "Set VELOX_GPU_CACHE_FULL_CAPACITY_TEST on an idle 96 GiB GPU";
  }
  constexpr int64_t kGiB = 1L << 30;
  size_t initialFree = 0, total = 0;
  GpuEncodedCache::check(cudaMemGetInfo(&initialFree, &total));
  ASSERT_GE(initialFree, 92L * kGiB);
  auto resource = createCudfCustomMemoryResource(90L * kGiB, true);
  rmm::mr::statistics_resource_adaptor tracker(createMemoryResource("cuda", 0));
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream{tracker};
  auto cache = std::make_shared<GpuEncodedCache>(resource, upstream);
  for (int i = 0; i < 64; ++i) {
    ASSERT_TRUE(cache->getOrFill(std::to_string(i), kGiB, nullptr, [](void* p) {
      GpuEncodedCache::check(cudaMemsetAsync(p, 0x5a, kGiB));
    }));
  }
  auto root = memory::memoryManager()->addCustomRootPool(
      "full-device-demand", resource);
  auto leaf = root->addLeafChild("request");
  CudfMemoryResource allocation(upstream, leaf);
  auto* data = allocation.allocate_sync(32L * kGiB, alignof(std::max_align_t));
  const auto counts = tracker.get_bytes_counter();
  size_t currentFree = 0;
  GpuEncodedCache::check(cudaMemGetInfo(&currentFree, &total));
  std::cout << "GPU_CACHE_CAPACITY live=" << counts.value
            << " peak=" << counts.peak
            << " device_used_delta=" << initialFree - currentFree
            << " cache_freed=" << cache->snapshot()["freed_bytes"].asInt()
            << std::endl;
  EXPECT_GE(counts.peak, 88L * kGiB);
  EXPECT_LE(counts.peak, 90L * kGiB);
  EXPECT_EQ(resource->arbitrator()->stats().numAborted, 0);
  allocation.deallocate_sync(data, 32L * kGiB, alignof(std::max_align_t));
  cache->evict(0);
  EXPECT_EQ(tracker.get_bytes_counter().value, 0);
}

} // namespace
} // namespace facebook::velox::cudf_velox
