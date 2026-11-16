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

#include "velox/common/file/File.h"

#include <rmm/cuda_stream.hpp>

#include <folly/ScopeGuard.h>
#include <folly/executors/InlineExecutor.h>
#include <gtest/gtest.h>

#include <future>
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

TEST_F(GpuInputCacheTest, ShardedAdmissionAtParticipantLimitBypasses) {
  setCudfContextDevice(0);
  auto owner = createCudfCustomMemoryResource(64L << 20, true);
  auto cache =
      std::make_shared<GpuEncodedCache>(owner, createMemoryResource("cuda", 0));
  auto first = cache->getOrFill("retained", 56L << 20, nullptr, [](void* p) {
    GpuEncodedCache::check(cudaMemsetAsync(p, 0x5a, 56L << 20));
  });
  ASSERT_TRUE(first);
  first
      .reset(); // Even evictable entries must not be reclaimed to admit a fill.
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
      bool cacheable)
      : BufferedInput(std::move(file), pool), cacheable_(cacheable) {}
  folly::Executor* executor() const override {
    return &folly::InlineExecutor::instance();
  }
  bool hasCache() const override {
    return cacheable_;
  }

 private:
  bool cacheable_;
};

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

} // namespace
} // namespace facebook::velox::cudf_velox
