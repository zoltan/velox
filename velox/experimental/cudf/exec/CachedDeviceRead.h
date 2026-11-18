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

#include "velox/experimental/cudf/exec/GpuInputCache.h"
#include "velox/experimental/gpu/DeviceReadSource.h"

#include <limits>

namespace facebook::velox::cudf_velox {

/// Describe whether an encoded range was retained, reused or refused.
enum class DeviceCacheOutcome { Hit, Fill, Bypass };

/// Pair a device-input lease with its cache disposition.
struct CachedDeviceRead {
  gpu::DeviceReadBatch batch;
  DeviceCacheOutcome outcome{DeviceCacheOutcome::Bypass};
};

/// Format-independent cache admission and lease-to-device-input adaptation.
/// Padding belongs to the key/byte contract supplied by the format adapter.
/// Return bytes as the logical view length; retain bytes + padding in storage.
/// A supplied lease must match that allocation size.
/// A rejected admission returns an empty batch; the caller retains its existing
/// allocation/coalescing policy for bypass. fill must enqueue on producer and
/// retain host sources until DMA completes, just as GpuInputCache requires.
/// An optional prior lookup lets batch readers prefetch only missing ranges.
CachedDeviceRead readCachedDeviceRange(
    const std::shared_ptr<GpuInputCache>& cache,
    const std::string& key,
    uint64_t bytes,
    uint64_t padding,
    cuda::stream_ref producer,
    const std::function<void(void*)>& fill,
    std::shared_ptr<const void> identity = nullptr,
    GpuInputCache::LeasePtr lease = nullptr);

} // namespace facebook::velox::cudf_velox
