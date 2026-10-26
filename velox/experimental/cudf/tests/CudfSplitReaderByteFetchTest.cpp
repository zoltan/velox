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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderByteFetch.h"

#include "velox/common/base/Exceptions.h"
#include "velox/common/base/tests/GTestUtils.h"

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <future>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

// Models a datasource that implements only caller-allocated device reads.
class CopyDeviceSource : public cudf::io::datasource {
 public:
  explicit CopyDeviceSource(std::shared_ptr<rmm::device_buffer> allocation)
      : allocation_{std::move(allocation)} {}

  std::unique_ptr<buffer> host_read(size_t, size_t) override {
    CUDF_FAIL("Unexpected host read");
  }

  size_t host_read(size_t, size_t, uint8_t*) override {
    CUDF_FAIL("Unexpected host read");
  }

  bool supports_device_read() const override {
    return true;
  }

  size_t size() const override {
    return allocation_->size();
  }

  std::future<size_t> device_read_async(
      size_t offset,
      size_t length,
      uint8_t* destination,
      cuda::stream_ref stream) override {
    ++numCopiedReads;
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        destination,
        static_cast<const uint8_t*>(allocation_->data()) + offset,
        length,
        cudaMemcpyDeviceToDevice,
        stream.get()));
    return std::async(std::launch::deferred, [stream, length] {
      stream.sync();
      return length;
    });
  }

  std::atomic<size_t> numCopiedReads{0};

 protected:
  std::shared_ptr<rmm::device_buffer> allocation_;
};

// Pins the source allocation while exposing just the requested range.
class ResidentRange final : public cudf::io::datasource::buffer {
 public:
  ResidentRange(
      std::shared_ptr<rmm::device_buffer> allocation,
      size_t offset,
      size_t length)
      : allocation_{std::move(allocation)}, offset_{offset}, length_{length} {}

  const uint8_t* data() const override {
    return static_cast<const uint8_t*>(allocation_->data()) + offset_;
  }

  size_t size() const override {
    return length_;
  }

 private:
  std::shared_ptr<rmm::device_buffer> allocation_;
  const size_t offset_;
  const size_t length_;
};

// Models cached bytes with both owning and caller-allocated read interfaces.
class ResidentDeviceSource : public CopyDeviceSource,
                             public gpu::DeviceReadSource {
 public:
  using CopyDeviceSource::CopyDeviceSource;
  using CopyDeviceSource::device_read;

  std::unique_ptr<buffer> device_read(
      size_t offset,
      size_t length,
      cuda::stream_ref /*stream*/) override {
    ++numOwnedReads;
    return std::make_unique<ResidentRange>(allocation_, offset, length);
  }

  gpu::DeviceReadBatch readDeviceRangesAsync(
      std::span<const gpu::DeviceByteRange> ranges,
      bool /*serializeSubmissions*/,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref /*outputResource*/,
      rmm::device_async_resource_ref /*temporaryResource*/) override {
    auto buffers = std::make_shared<
        std::vector<std::unique_ptr<cudf::io::datasource::buffer>>>();
    gpu::DeviceReadBatch result;
    result.owner = buffers;
    for (const auto& range : ranges) {
      auto buffer = device_read(range.offset(), range.size(), stream);
      result.data.emplace_back(buffer->data(), buffer->size());
      buffers->push_back(std::move(buffer));
    }
    return result;
  }

  std::atomic<size_t> numOwnedReads{0};
};

// Exercises the same datasource allocation path used by BufferedInput.
class AllocatingDeviceSource final
    : public CopyDeviceSource,
      public gpu::DeviceReadSource,
      public std::enable_shared_from_this<AllocatingDeviceSource> {
 public:
  using CopyDeviceSource::CopyDeviceSource;

  gpu::DeviceReadBatch readDeviceRangesAsync(
      std::span<const gpu::DeviceByteRange> ranges,
      bool serializeSubmissions,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref outputResource,
      rmm::device_async_resource_ref temporaryResource) override {
    ++numBatchReads;
    std::vector<cudf::io::text::byte_range_info> byteRanges;
    for (const auto& range : ranges) {
      byteRanges.emplace_back(range.offset(), range.size());
    }
    return copyByteRangesAsync(
        shared_from_this(),
        {byteRanges.data(), byteRanges.size()},
        serializeSubmissions,
        stream,
        {outputResource, temporaryResource});
  }

  size_t numBatchReads{0};
};

class CudfSplitReaderByteFetchTest : public testing::Test {
 protected:
  void SetUp() override {
    std::iota(input_.begin(), input_.end(), uint8_t{0});
  }

  // Keep cache population outside the consumer's allocation accounting.
  std::shared_ptr<rmm::device_buffer> populate() {
    auto allocation = std::make_shared<rmm::device_buffer>(
        input_.data(), input_.size(), stream_, resource_);
    stream_.synchronize();
    return allocation;
  }

  void verify(gpu::DeviceByteView data, cudf::io::text::byte_range_info range) {
    ASSERT_EQ(data.size(), range.size());
    std::vector<uint8_t> actual(data.size());
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        actual.data(),
        data.data(),
        data.size(),
        cudaMemcpyDeviceToHost,
        stream_.value()));
    stream_.synchronize();
    EXPECT_THAT(
        actual,
        testing::ElementsAreArray(
            input_.begin() + range.offset(),
            input_.begin() + range.offset() + range.size()));
  }

  rmm::cuda_stream stream_;
  rmm::mr::cuda_memory_resource resource_;
  std::vector<uint8_t> input_ = std::vector<uint8_t>(4'096);
};

TEST_F(CudfSplitReaderByteFetchTest, datasourceReturnsResidentBytes) {
  auto allocation = populate();
  std::weak_ptr<rmm::device_buffer> lifetime = allocation;
  auto source = std::make_shared<ResidentDeviceSource>(allocation);
  const cudf::io::text::byte_range_info range{19, 37};
  auto buffer = source->device_read(range.offset(), range.size(), stream_);
  EXPECT_EQ(
      buffer->data(),
      static_cast<const uint8_t*>(allocation->data()) + range.offset());
  allocation.reset();
  source.reset();
  EXPECT_FALSE(lifetime.expired());
  verify({buffer->data(), buffer->size()}, range);
  buffer.reset();
  EXPECT_TRUE(lifetime.expired());
}

TEST_F(CudfSplitReaderByteFetchTest, fetchReturnsResidentBytes) {
  for (const bool serialize : {false, true}) {
    SCOPED_TRACE(serialize);
    auto allocation = populate();
    auto source = std::make_shared<ResidentDeviceSource>(allocation);
    rmm::mr::statistics_resource_adaptor allocations{resource_};
    const std::vector<cudf::io::text::byte_range_info> ranges{
        {19, 37}, {91, 48}, {19, 37}};
    auto fetch = fetchByteRangesAsync(
        source,
        {ranges.data(), ranges.size()},
        serialize,
        stream_,
        cudf::memory_resources{allocations, allocations});
    fetch.wait();
    ASSERT_THAT(fetch.data, testing::SizeIs(ranges.size()));
    for (size_t i = 0; i < ranges.size(); ++i) {
      SCOPED_TRACE(i);
      EXPECT_EQ(
          fetch.data[i].data(),
          static_cast<const uint8_t*>(allocation->data()) + ranges[i].offset());
      verify(fetch.data[i], ranges[i]);
    }
    EXPECT_GT(source->numOwnedReads.load(), 0);
    EXPECT_EQ(source->numCopiedReads.load(), 0);
    EXPECT_EQ(allocations.get_bytes_counter().total, 0);
  }
}

TEST_F(CudfSplitReaderByteFetchTest, fetchRetainsResidentBytes) {
  for (const bool serialize : {false, true}) {
    SCOPED_TRACE(serialize);
    auto allocation = populate();
    std::weak_ptr<rmm::device_buffer> lifetime = allocation;
    auto source = std::make_shared<ResidentDeviceSource>(allocation);
    const cudf::io::text::byte_range_info range{91, 48};
    {
      auto fetch = fetchByteRangesAsync(
          source,
          {&range, 1},
          serialize,
          stream_,
          cudf::memory_resources{resource_, resource_});
      fetch.wait();
      allocation.reset();
      source.reset();
      EXPECT_FALSE(lifetime.expired());
      ASSERT_THAT(fetch.data, testing::SizeIs(1));
      verify(fetch.data.front(), range);
    }
    EXPECT_TRUE(lifetime.expired());
  }
}

TEST_F(CudfSplitReaderByteFetchTest, copyOnlySourceRemainsSupported) {
  for (const bool serialize : {false, true}) {
    SCOPED_TRACE(serialize);
    auto source = std::make_shared<CopyDeviceSource>(populate());
    const cudf::io::text::byte_range_info range{19, 37};
    auto fetch = fetchByteRangesAsync(
        source,
        {&range, 1},
        serialize,
        stream_,
        cudf::memory_resources{resource_, resource_});
    fetch.wait();
    EXPECT_GT(source->numCopiedReads.load(), 0);
    source.reset();
    ASSERT_THAT(fetch.data, testing::SizeIs(1));
    verify(fetch.data.front(), range);
  }
}

TEST_F(CudfSplitReaderByteFetchTest, allocationAndCoalescingMatchCopyPath) {
  for (const bool serialize : {false, true}) {
    SCOPED_TRACE(serialize);
    auto allocation = populate();
    auto baseline = std::make_shared<CopyDeviceSource>(allocation);
    auto source = std::make_shared<AllocatingDeviceSource>(allocation);
    const std::vector<cudf::io::text::byte_range_info> ranges{
        {19, 37}, {56, 48}, {200, 17}};
    auto read = [&](const auto& dataSource) {
      rmm::mr::statistics_resource_adaptor allocations{resource_};
      int64_t totalBytes{0};
      {
        auto fetch = fetchByteRangesAsync(
            dataSource,
            {ranges.data(), ranges.size()},
            serialize,
            stream_,
            {allocations, allocations});
        fetch.waitOn(stream_);
        ASSERT_THAT(fetch.data, testing::SizeIs(ranges.size()));
        for (size_t i = 0; i < ranges.size(); ++i) {
          verify(fetch.data[i], ranges[i]);
        }
        totalBytes = allocations.get_bytes_counter().total;
        EXPECT_EQ(totalBytes, 104);
        EXPECT_EQ(allocations.get_allocations_counter().total, 1);
      }
      EXPECT_EQ(allocations.get_bytes_counter().value, 0);
    };
    read(baseline);
    read(source);
    EXPECT_EQ(source->numBatchReads, 1);
    EXPECT_EQ(baseline->numCopiedReads.load(), 2);
    EXPECT_EQ(source->numCopiedReads.load(), 2);
  }
}

TEST_F(CudfSplitReaderByteFetchTest, rejectsMalformedOwnedBatches) {
  // Corrupt the provider's result without changing its production lifetime.
  class MalformedSource final : public ResidentDeviceSource {
   public:
    enum class Failure { kOwner, kCount, kSize, kPointer };

    MalformedSource(
        std::shared_ptr<rmm::device_buffer> allocation,
        Failure failure)
        : ResidentDeviceSource(std::move(allocation)), failure_{failure} {}

    gpu::DeviceReadBatch readDeviceRangesAsync(
        std::span<const gpu::DeviceByteRange> ranges,
        bool serializeSubmissions,
        cuda::stream_ref stream,
        rmm::device_async_resource_ref outputResource,
        rmm::device_async_resource_ref temporaryResource) override {
      auto batch = ResidentDeviceSource::readDeviceRangesAsync(
          ranges,
          serializeSubmissions,
          stream,
          outputResource,
          temporaryResource);
      switch (failure_) {
        case Failure::kOwner:
          batch.owner.reset();
          break;
        case Failure::kCount:
          batch.data.clear();
          break;
        case Failure::kSize:
          batch.data.front() = {batch.data.front().data(), 1};
          break;
        case Failure::kPointer:
          batch.data.front() = {nullptr, batch.data.front().size()};
          break;
      }
      return batch;
    }

   private:
    const Failure failure_;
  };
  using Failure = MalformedSource::Failure;
  const std::vector<std::pair<Failure, std::string>> cases{
      {Failure::kOwner, "Missing device range owner"},
      {Failure::kCount, "Device range count mismatch"},
      {Failure::kSize, "Device range size mismatch"},
      {Failure::kPointer, "Null device range data"}};
  const cudf::io::text::byte_range_info range{19, 37};
  for (const auto& [failure, message] : cases) {
    SCOPED_TRACE(message);
    auto source = std::make_shared<MalformedSource>(populate(), failure);
    VELOX_ASSERT_THROW(
        fetchByteRangesAsync(
            source, {&range, 1}, false, stream_, {resource_, resource_}),
        message);
  }
}

TEST_F(CudfSplitReaderByteFetchTest, invalidResidentRangesFailBeforeRead) {
  auto source = std::make_shared<ResidentDeviceSource>(populate());
  const std::vector<std::pair<cudf::io::text::byte_range_info, std::string>>
      cases{
          {{4'097, 0}, "Byte range exceeds datasource size"},
          {{4'095, 2}, "Byte range exceeds datasource size"}};
  for (const auto& [range, message] : cases) {
    SCOPED_TRACE(message);
    VELOX_ASSERT_THROW(
        fetchByteRangesAsync(
            source, {&range, 1}, false, stream_, {resource_, resource_}),
        message);
  }
  EXPECT_EQ(source->numOwnedReads.load(), 0);
  EXPECT_EQ(source->numCopiedReads.load(), 0);
}

TEST_F(CudfSplitReaderByteFetchTest, emptyResidentBatchIsStarted) {
  auto source = std::make_shared<ResidentDeviceSource>(populate());
  auto fetch =
      fetchByteRangesAsync(source, {}, false, stream_, {resource_, resource_});
  EXPECT_TRUE(static_cast<bool>(fetch));
  EXPECT_FALSE(fetch.pending.valid());
  EXPECT_THAT(fetch.data, testing::IsEmpty());
  fetch.waitOn(stream_);
  fetch = {};
  EXPECT_FALSE(static_cast<bool>(fetch));
}

TEST_F(CudfSplitReaderByteFetchTest, retainsSourceBeforeWaiting) {
  for (const bool resident : {false, true}) {
    SCOPED_TRACE(resident);
    std::shared_ptr<CopyDeviceSource> source;
    if (resident) {
      source = std::make_shared<ResidentDeviceSource>(populate());
    } else {
      source = std::make_shared<CopyDeviceSource>(populate());
    }
    std::weak_ptr<CopyDeviceSource> lifetime = source;
    const cudf::io::text::byte_range_info range{19, 37};
    auto fetch = fetchByteRangesAsync(
        source, {&range, 1}, false, stream_, {resource_, resource_});
    source.reset();
    EXPECT_FALSE(lifetime.expired());
    fetch.waitOn(stream_);
    verify(fetch.data.front(), range);
    fetch = {};
    EXPECT_TRUE(lifetime.expired());
  }
}

TEST_F(CudfSplitReaderByteFetchTest, finishesSubmissionBeforeRelease) {
  for (const bool replace : {false, true}) {
    SCOPED_TRACE(replace);
    bool submitted{false};
    auto allocation = populate();
    std::weak_ptr<rmm::device_buffer> lifetime = allocation;
    {
      gpu::DeviceReadBatch batch;
      batch.owner = std::move(allocation);
      batch.pending = std::async(std::launch::deferred, [&] {
        EXPECT_FALSE(lifetime.expired());
        submitted = true;
      });
      if (replace) {
        batch = {};
        EXPECT_TRUE(submitted);
        EXPECT_TRUE(lifetime.expired());
      }
    }
    EXPECT_TRUE(submitted);
    EXPECT_TRUE(lifetime.expired());
  }
}

TEST_F(CudfSplitReaderByteFetchTest, waitPropagatesReadFailure) {
  gpu::DeviceReadBatch batch;
  batch.owner = populate();
  batch.pending = std::async(std::launch::deferred, [] {
    throw std::runtime_error("Read failed before submission");
  });
  try {
    batch.wait();
    FAIL() << "Expected read failure";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "Read failed before submission");
  }
  EXPECT_FALSE(batch.pending.valid());
  batch.wait();
}

TEST_F(CudfSplitReaderByteFetchTest, ordersFirstConsumerAfterProduction) {
  gpu::DeviceReadBatch batch;
  batch.owner = populate();
  bool submitted{false};
  bool ready{false};
  batch.pending = std::async(std::launch::deferred, [&] { submitted = true; });
  batch.ready = [&](cuda::stream_ref consumer) {
    EXPECT_TRUE(submitted);
    EXPECT_EQ(consumer.get(), stream_.value());
    ready = true;
  };
  batch.waitOn(stream_);
  EXPECT_TRUE(ready);
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
