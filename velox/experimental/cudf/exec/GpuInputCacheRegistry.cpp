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
#include "velox/experimental/cudf/exec/GpuInputCache.h"
#include "velox/experimental/cudf/exec/GpuInputCacheRegistry.h"

#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/memory/SharedArbitrator.h"

#include <mutex>
namespace facebook::velox::cudf_velox {
namespace {
std::mutex cacheMutex;
std::weak_ptr<GpuInputCache> processCache;
std::shared_ptr<GpuInputCache> registeredCache;
} // namespace
void setGpuInputCache(const std::shared_ptr<GpuInputCache>& cache) {
  std::lock_guard lock(cacheMutex);
  processCache = cache;
}
void initializeGpuInputCache(bool enabled) {
  auto resource = cudfCustomMemoryResource();
  if (!enabled || !cache::AsyncDataCache::getInstance() || !resource || !mr_) {
    return;
  }
  auto* arbitrator =
      dynamic_cast<memory::SharedArbitrator*>(resource->arbitrator());
  if (!arbitrator || !arbitrator->discardableReclaimEnabled()) {
    return;
  }
  std::lock_guard lock(cacheMutex);
  if (!processCache.expired()) {
    return;
  }
  registeredCache = GpuInputCache::create(resource, *mr_);
  processCache = registeredCache;
}

void resetGpuInputCache() {
  std::shared_ptr<GpuInputCache> previous;
  {
    std::lock_guard lock(cacheMutex);
    processCache.reset();
    previous = std::move(registeredCache);
  }
  // Destruction may wait for CUDA events; do it outside the registry lock.
  previous.reset();
}

std::shared_ptr<GpuInputCache> gpuInputCache() {
  std::lock_guard lock(cacheMutex);
  return processCache.lock();
}
} // namespace facebook::velox::cudf_velox
