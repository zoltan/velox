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

#include "velox/experimental/gpu/DeviceReadSource.h"

#include <cudf/io/datasource.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <future>
#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

/// Owns source-provided input and its pending reads.
using ByteRangeFetch = gpu::DeviceReadBatch;

/// Fetches source-owned bytes when supported, or uses cuDF's copy path.
/// Retains the source until pending work completes. Providers must supply
/// backing storage padded for cuDF's Parquet decoder (8-byte allocation tail
/// alignment), matching fetch_byte_ranges_to_device_async. Consume on the
/// supplied stream and retain the batch until the last use is submitted.
ByteRangeFetch fetchByteRangesAsync(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    bool serializeSubmissions,
    cuda::stream_ref stream,
    cudf::memory_resources mr);

/// Allocates and fills device input with cuDF's existing batched copy
/// algorithm. Bypasses ownership dispatch so datasource implementations can
/// reuse it.
ByteRangeFetch copyByteRangesAsync(
    std::shared_ptr<cudf::io::datasource> dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> byteRanges,
    bool serializeSubmissions,
    cuda::stream_ref stream,
    cudf::memory_resources mr);

// Reads page-index ranges into host buffers and returns them with their views.
std::pair<
    std::vector<std::unique_ptr<cudf::io::datasource::buffer>>,
    std::vector<cudf::host_span<const uint8_t>>>
fetchPageIndexes(
    const std::shared_ptr<cudf::io::datasource>& dataSource,
    cudf::host_span<const cudf::io::text::byte_range_info> pageIndexByteRanges);

} // namespace facebook::velox::cudf_velox::connector::hive
