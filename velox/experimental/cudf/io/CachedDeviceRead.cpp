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
#include "velox/experimental/cudf/io/CachedDeviceRead.h"

namespace facebook::velox::cudf_velox {
CachedDeviceRead readCachedDeviceRange(
    const std::shared_ptr<GpuEncodedCache>& cache,
    const std::string& key,
    uint64_t bytes,
    uint64_t padding,
    cuda::stream_ref producer,
    const std::function<void(void*)>& fill,
    std::shared_ptr<const void> identity,
    GpuEncodedCache::LeasePtr lease) {
  VELOX_CHECK_LE(padding, std::numeric_limits<uint64_t>::max() - bytes);
  bool filled = false;
  if (!lease && cache) {
    lease = cache->getOrFill(
        key,
        bytes + padding,
        producer.get(),
        [&](void* dst) {
          if (padding) {
            VELOX_CHECK(
                cudaMemsetAsync(
                    static_cast<char*>(dst) + bytes,
                    0,
                    padding,
                    producer.get()) == cudaSuccess);
          }
          fill(dst);
          filled = true;
        },
        std::move(identity));
  }
  CachedDeviceRead result;
  if (!lease)
    return result;
  VELOX_CHECK_EQ(
      lease->size(),
      bytes + padding,
      "Cached allocation size does not match requested bytes and padding");
  result.outcome = filled ? DeviceCacheOutcome::Fill : DeviceCacheOutcome::Hit;
  result.batch.data.emplace_back(
      static_cast<const uint8_t*>(lease->data()), bytes);
  result.batch.owner = lease;
  result.batch.ready = [lease = std::move(lease)](cuda::stream_ref consumer) {
    lease->waitForFillOn(consumer.get());
  };
  return result;
}
} // namespace facebook::velox::cudf_velox
