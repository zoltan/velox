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
#include "velox/experimental/cudf/connectors/hive/ScopedReadSuspension.h"

#include "velox/common/base/Exceptions.h"

#include <cudf/io/parquet_io_utils.hpp>

#include <algorithm>
#include <functional>
#include <iterator>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

std::pair<
    std::vector<std::unique_ptr<cudf::io::datasource::buffer>>,
    std::vector<cudf::host_span<const uint8_t>>>
fetchPageIndexes(
    const std::shared_ptr<cudf::io::datasource>& dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info>
        pageIndexByteRanges) {
  std::vector<std::reference_wrapper<cudf::io::datasource>> dataSources{
      std::ref(*dataSource)};
  auto buffers = cudf::io::parquet::fetch_page_indexes_to_host(
      dataSources, pageIndexByteRanges);

  std::vector<cudf::host_span<const uint8_t>> spans;
  spans.reserve(buffers.size());
  std::transform(
      buffers.begin(),
      buffers.end(),
      std::back_inserter(spans),
      [](const auto& buffer) {
        return buffer->size() == 0 ? cudf::host_span<const uint8_t>{}
                                   : cudf::host_span<const uint8_t>{*buffer};
      });

  return {std::move(buffers), std::move(spans)};
}

ByteRangeFetch fetchByteRangesAsync(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    bool serializeSubmissions,
    cuda::stream_ref stream,
    cudf::memory_resources mr) {
  ScopedReadSuspension suspension;
  auto* source = dynamic_cast<gpu::DeviceReadSource*>(dataSource.get());
  if (source == nullptr) {
    return copyByteRangesAsync(
        std::move(dataSource), byteRanges, serializeSubmissions, stream, mr);
  }

  std::vector<gpu::DeviceByteRange> ranges;
  ranges.reserve(byteRanges.size());
  for (const auto& range : byteRanges) {
    VELOX_CHECK_LE(
        static_cast<uint64_t>(range.offset()),
        dataSource->size(),
        "Byte range exceeds datasource size");
    VELOX_CHECK_LE(
        static_cast<uint64_t>(range.size()),
        dataSource->size() - range.offset(),
        "Byte range exceeds datasource size");
    ranges.emplace_back(range.offset(), range.size());
  }
  auto result = source->readDeviceRangesAsync(
      ranges,
      serializeSubmissions,
      stream,
      mr.get_output_mr(),
      mr.get_temporary_mr());
  VELOX_CHECK_NOT_NULL(result.owner, "Missing device range owner");
  struct Owner {
    std::shared_ptr<cudf::io::datasource> source;
    std::shared_ptr<void> storage;
  };
  auto owner = std::make_shared<Owner>(Owner{dataSource, result.owner});
  // Retain source state through completion, including failed submissions.
  result.owner = std::move(owner);
  VELOX_CHECK_EQ(
      result.data.size(), ranges.size(), "Device range count mismatch");
  for (size_t i = 0; i < ranges.size(); ++i) {
    VELOX_CHECK_EQ(
        result.data[i].size(),
        static_cast<size_t>(ranges[i].size()),
        "Device range size mismatch");
    if (!result.data[i].empty()) {
      VELOX_CHECK_NOT_NULL(result.data[i].data(), "Null device range data");
    }
  }
  return result;
}

ByteRangeFetch copyByteRangesAsync(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    bool serializeSubmissions,
    cuda::stream_ref stream,
    cudf::memory_resources mr) {
  ScopedReadSuspension suspension;
  // Allocate the owner before starting I/O so host allocation failure cannot
  // release buffers while reads are still pending.
  struct Owner {
    std::shared_ptr<cudf::io::datasource> source;
    std::vector<rmm::device_buffer> buffers;
    std::future<void> pending;
    ~Owner() {
      if (pending.valid()) {
        ScopedReadSuspension suspension;
        pending.wait();
      }
    }
  };
  auto owner = std::make_shared<Owner>();
  owner->source = std::move(dataSource);
  auto const ioSubmissionPolicy = serializeSubmissions
      ? cudf::io::parquet::io_submission_policy::SERIALIZE
      : cudf::io::parquet::io_submission_policy::INTERLEAVE;
  auto [buffers, data, pending] =
      cudf::io::parquet::fetch_byte_ranges_to_device_async(
          *owner->source,
          {byteRanges.data(), byteRanges.size()},
          ioSubmissionPolicy,
          stream,
          mr);
  ByteRangeFetch result;
  result.owner = owner;
  owner->buffers = std::move(buffers);
  owner->pending = std::move(pending);
  if (owner->pending.valid()) {
    result.pending = std::async(std::launch::deferred, [owner] {
      ScopedReadSuspension suspension;
      owner->pending.get();
    });
  }
  result.data.reserve(data.size());
  for (const auto& view : data) {
    result.data.emplace_back(view.data(), view.size());
  }
  return result;
}

} // namespace facebook::velox::cudf_velox::connector::hive
