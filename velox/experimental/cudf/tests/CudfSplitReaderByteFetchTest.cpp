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

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <future>
#include <memory>
#include <numeric>
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
class ResidentDeviceSource final : public CopyDeviceSource {
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

  std::atomic<size_t> numOwnedReads{0};
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

  void verify(
      cudf::device_span<const uint8_t> data,
      cudf::io::text::byte_range_info range) {
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

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
