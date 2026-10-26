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

#include "velox/experimental/gpu/DeviceReadSource.h"

namespace facebook::velox::gpu {

void DeviceReadBatch::wait() {
  if (pending.valid()) {
    pending.get();
  }
}

void DeviceReadBatch::waitOn(cuda::stream_ref stream) {
  wait();
  if (ready) {
    ready(stream);
  }
}

DeviceReadBatch::~DeviceReadBatch() {
  if (pending.valid()) {
    // wait() also runs deferred work, but leaves its exception in the future.
    // Consumers observe read failures through the explicit wait() method.
    pending.wait();
  }
}

DeviceReadBatch& DeviceReadBatch::operator=(DeviceReadBatch&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  DeviceReadBatch previous(std::move(*this));
  owner = std::move(other.owner);
  data = std::move(other.data);
  pending = std::move(other.pending);
  ready = std::move(other.ready);
  return *this;
}

} // namespace facebook::velox::gpu
