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

#pragma once

#include "velox/experimental/cudf/exec/GpuInputCacheRegistry.h"
#include "velox/experimental/gpu/DeviceReadSource.h"

#include "velox/common/caching/StringIdMap.h"
#include "velox/dwio/common/BufferedInput.h"

#include <cudf/io/datasource.hpp>

#include <cuda/stream>

#include <future>
#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

/// Drains accepted host uploads before unregistering cuDF resources. Call after
/// stopping reader submissions; concurrent shutdown/submission is unsupported.
/// A later read lazily starts a new retirement worker.
void drainInputUploads();

// A cudf::io::datasource that serves bytes via Velox BufferedInput so that
// reads benefit from AsyncDataCache / SSD cache and are always returned as
// contiguous buffers.
class BufferedInputDataSource
    : public cudf::io::datasource,
      public gpu::DeviceReadSource,
      public std::enable_shared_from_this<BufferedInputDataSource> {
 public:
  explicit BufferedInputDataSource(
      std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input,
      std::shared_ptr<IoStats> ioStats = nullptr,
      std::shared_ptr<const StringIdLease> fileIdentity = nullptr);

  [[nodiscard]] size_t size() const override;

  std::unique_ptr<datasource::buffer> host_read(size_t offset, size_t size)
      override;

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  std::future<std::unique_ptr<datasource::buffer>> host_read_async(
      size_t offset,
      size_t size) override;

  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst);

  [[nodiscard]] bool supports_device_read() const override;

  std::future<size_t> device_read_async(
      size_t offset,
      size_t size,
      uint8_t* dst,
      cuda::stream_ref stream) override;

  /// Returns datasource-owned input using the existing batched copy algorithm.
  gpu::DeviceReadBatch readDeviceRangesAsync(
      std::span<const gpu::DeviceByteRange> ranges,
      bool serializeSubmissions,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref outputResource,
      rmm::device_async_resource_ref temporaryResource) override;

 private:
  void readContiguous(size_t offset, size_t size, uint8_t* dst);

  // Return leased cache input, using the existing batched copy for bypasses.
  gpu::DeviceReadBatch readCachedRanges(
      std::span<const gpu::DeviceByteRange> ranges,
      bool serializeSubmissions,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref outputResource,
      rmm::device_async_resource_ref temporaryResource);

  void recordBytes(const char* name, uint64_t bytes);

  std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input_;
  const size_t fileSize_;
  std::shared_ptr<IoStats> ioStats_;
  // Keep the immutable file generation alive even when host bytes are evicted.
  std::shared_ptr<const StringIdLease> fileIdentity_;
  std::shared_ptr<GpuInputCache> deviceCache_;
};

} // namespace facebook::velox::cudf_velox::connector::hive
