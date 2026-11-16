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
#include "velox/experimental/cudf/exec/EncodedInputGpuCache.h"

#include <mutex>
namespace facebook::velox::cudf_velox {
namespace {
std::mutex cacheMutex;
std::weak_ptr<GpuEncodedCache> processCache;
} // namespace
void setEncodedInputGpuCache(const std::shared_ptr<GpuEncodedCache>& cache) {
  std::lock_guard lock(cacheMutex);
  processCache = cache;
}
std::shared_ptr<GpuEncodedCache> encodedInputGpuCache() {
  std::lock_guard lock(cacheMutex);
  return processCache.lock();
}
} // namespace facebook::velox::cudf_velox
