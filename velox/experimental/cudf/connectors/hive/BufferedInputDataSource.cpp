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
#include "velox/experimental/cudf/connectors/hive/BufferedInputDataSource.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderByteFetch.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/io/CachedDeviceRead.h"

#include "velox/common/base/Exceptions.h"

#include <folly/Executor.h>

#include <algorithm>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

namespace {

// Launches the `read` task on the IO executor or waiting thread, whichever
// starts first.
std::future<size_t> submitDeviceRead(
    folly::Executor* executor,
    std::function<size_t()> read) {
  auto task = std::make_shared<std::packaged_task<size_t()>>(std::move(read));
  auto result = task->get_future();
  auto once = std::make_shared<std::once_flag>();
  auto run = [once, task]() { std::call_once(*once, [task]() { (*task)(); }); };
  // Complete fallible future construction before the executor can accept work.
  auto completion = std::async(
      std::launch::deferred, [run, result = std::move(result)]() mutable {
        run();
        return result.get();
      });
  executor->add(std::move(run));
  return completion;
}

} // namespace

BufferedInputDataSource::BufferedInputDataSource(
    std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input,
    std::shared_ptr<IoStats> ioStats,
    std::shared_ptr<const StringIdLease> fileIdentity)
    : input_(std::move(input)),
      fileSize_(input_->getReadFile()->size()),
      ioStats_(std::move(ioStats)),
      fileIdentity_(std::move(fileIdentity)),
      deviceCache_(
          input_->hasCache() && fileIdentity_ ? encodedInputGpuCache()
                                              : nullptr) {}

void BufferedInputDataSource::recordBytes(const char* name, uint64_t bytes) {
  if (ioStats_) {
    ioStats_->addCounter(
        name, RuntimeCounter(bytes, RuntimeCounter::Unit::kBytes));
  }
}

size_t BufferedInputDataSource::size() const {
  return fileSize_;
}

std::unique_ptr<cudf::io::datasource::buffer>
BufferedInputDataSource::host_read(size_t offset, size_t size) {
  if (offset >= fileSize_) {
    return cudf::io::datasource::buffer::create(std::vector<uint8_t>{});
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  std::vector<uint8_t> data(readSize);
  readContiguous(offset, readSize, data.data());
  return cudf::io::datasource::buffer::create(std::move(data));
}

size_t
BufferedInputDataSource::host_read(size_t offset, size_t size, uint8_t* dst) {
  if (offset >= fileSize_) {
    return 0;
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  readContiguous(offset, readSize, dst);
  return readSize;
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>>
BufferedInputDataSource::host_read_async(size_t offset, size_t size) {
  return std::async(std::launch::deferred, [this, offset, size]() {
    return this->host_read(offset, size);
  });
}

std::future<size_t> BufferedInputDataSource::host_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  return std::async(std::launch::deferred, [this, offset, size, dst]() {
    return this->host_read(offset, size, dst);
  });
}

std::future<size_t> BufferedInputDataSource::device_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst,
    cuda::stream_ref stream) {
  VELOX_CHECK(input_->executor() != nullptr, "IO executor is not initialized");
  return submitDeviceRead(
      input_->executor(), [this, offset, size, dst, stream]() {
        auto hostBuffer = host_read(offset, size);
        ensureCudaContextForThread();
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            dst,
            hostBuffer->data(),
            hostBuffer->size(),
            cudaMemcpyDefault,
            stream.get()));
        recordBytes("cudfInputH2DBytes", hostBuffer->size());
        return hostBuffer->size();
      });
}

gpu::DeviceReadBatch BufferedInputDataSource::readDeviceRangesAsync(
    std::span<const gpu::DeviceByteRange> ranges,
    bool serializeSubmissions,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref outputResource,
    rmm::device_async_resource_ref temporaryResource) {
  if (deviceCache_) {
    return readCachedRanges(
        ranges,
        serializeSubmissions,
        stream,
        outputResource,
        temporaryResource);
  }
  std::vector<cudf::io::text::byte_range_info> byteRanges;
  byteRanges.reserve(ranges.size());
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

gpu::DeviceReadBatch BufferedInputDataSource::readCachedRanges(
    std::span<const gpu::DeviceByteRange> ranges,
    bool serializeSubmissions,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref outputResource,
    rmm::device_async_resource_ref temporaryResource) {
  struct Owner {
    std::shared_ptr<BufferedInputDataSource> source;
    std::vector<gpu::DeviceReadBatch> reads;
  };
  auto owner = std::make_shared<Owner>();
  owner->source = shared_from_this();
  owner->reads.reserve(ranges.size() + 1);
  gpu::DeviceReadBatch result;
  result.owner = owner;
  result.data.resize(ranges.size());
  std::vector<cudf::io::text::byte_range_info> bypass;
  std::vector<size_t> bypassIndices;
  bypass.reserve(ranges.size());
  bypassIndices.reserve(ranges.size());
  for (size_t i = 0; i < ranges.size(); ++i) {
    const auto& range = ranges[i];
    VELOX_CHECK_GE(range.offset(), 0);
    VELOX_CHECK_GE(range.size(), 0);
    VELOX_CHECK_LE(range.offset(), size());
    VELOX_CHECK_LE(range.size(), size() - range.offset());
    if (range.size() == 0) {
      continue;
    }
    recordBytes("cudfGpuCacheEligibleBytes", range.size());
    const auto key = fmt::format(
        "parquet:{}:{}:{}", fileIdentity_->id(), range.offset(), range.size());
    // Decoder loads can read the final 64-bit word past the logical tail.
    constexpr uint64_t kDecoderWordBytes = sizeof(uint64_t);
    const auto padding =
        (kDecoderWordBytes - range.size() % kDecoderWordBytes) %
        kDecoderWordBytes;
    auto cached = readCachedDeviceRange(
        deviceCache_,
        key,
        range.size(),
        padding,
        stream,
        [&](void* destination) {
          VELOX_CHECK_EQ(
              device_read_async(
                  range.offset(),
                  range.size(),
                  static_cast<uint8_t*>(destination),
                  stream)
                  .get(),
              range.size());
        },
        fileIdentity_);
    if (cached.batch) {
      result.data[i] = gpu::DeviceByteView{
          cached.batch.data.front().data(), static_cast<size_t>(range.size())};
      owner->reads.push_back(std::move(cached.batch));
      recordBytes(
          cached.outcome == DeviceCacheOutcome::Fill ? "cudfGpuCacheFillBytes"
                                                     : "cudfGpuCacheHitBytes",
          range.size());
    } else {
      bypass.emplace_back(range.offset(), range.size());
      bypassIndices.push_back(i);
      recordBytes("cudfGpuCacheBypassBytes", range.size());
    }
  }
  if (!bypass.empty()) {
    auto copied = copyByteRangesAsync(
        shared_from_this(),
        {bypass.data(), bypass.size()},
        serializeSubmissions,
        stream,
        {outputResource, temporaryResource});
    for (size_t i = 0; i < bypassIndices.size(); ++i) {
      result.data[bypassIndices[i]] = copied.data[i];
    }
    result.pending = std::move(copied.pending);
    owner->reads.push_back(std::move(copied));
  }
  // All views are consumed on the supplied stream, matching the copy adapter.
  return result;
}

bool BufferedInputDataSource::supports_device_read() const {
  return true;
}

void BufferedInputDataSource::readContiguous(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  using namespace facebook::velox::dwio::common;
  // BufferedInput::read gives us a stream over the exact region.
  auto stream = input_->read(offset, size, LogType::FILE);
  VELOX_CHECK(stream != nullptr, "read() returned null stream");
  stream->readFully(reinterpret_cast<char*>(dst), size);
}

} // namespace facebook::velox::cudf_velox::connector::hive
