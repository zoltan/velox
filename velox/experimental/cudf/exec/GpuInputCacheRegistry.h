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
#include <memory>
namespace facebook::velox::cudf_velox {
class GpuInputCache;
/// Install the process cache for immutable encoded input. The caller owns it.
/// Readers use it when CPU AsyncDataCache is also available.
void setGpuInputCache(const std::shared_ptr<GpuInputCache>& cache);
/// Install a registration-owned cache when host caching and an accounted GPU
/// backend are available. An explicitly installed cache takes precedence.
void initializeGpuInputCache(bool enabled);
/// Release registration ownership and unpublish the cache during cuDF shutdown.
/// Readers and queries must have finished before unregistering cuDF.
void resetGpuInputCache();
/// Return the installed cache, or nullptr if absent or no longer owned.
std::shared_ptr<GpuInputCache> gpuInputCache();
} // namespace facebook::velox::cudf_velox
