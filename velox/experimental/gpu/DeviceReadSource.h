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

#include <rmm/resource_ref.hpp>
#include <cuda/stream>

#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <span>
#include <vector>

namespace facebook::velox::gpu {

/// File byte coordinates. Signed so invalid/overflowing requests can be
/// rejected before I/O. This contract does not interpret the persisted
/// encoding.
class DeviceByteRange {
 public:
  /// Identifies an exact range in the encoded input.
  DeviceByteRange(int64_t offset, int64_t size)
      : offset_(offset), size_(size) {}
  /// Returns the byte offset from the start of the input.
  int64_t offset() const {
    return offset_;
  }
  /// Returns the requested number of bytes.
  int64_t size() const {
    return size_;
  }

 private:
  int64_t offset_;
  int64_t size_;
};

/// Non-owning device address: deliberately has no host dereference operator.
class DeviceByteView {
 public:
  /// Creates an empty device view.
  DeviceByteView() = default;
  /// Creates a view without taking ownership of its allocation.
  DeviceByteView(const uint8_t* data, size_t size) : data_(data), size_(size) {}
  /// Returns a device address, which must not be dereferenced on the host.
  const uint8_t* data() const {
    return data_;
  }
  /// Returns the logical size, excluding any decoder padding.
  size_t size() const {
    return size_;
  }
  /// Returns whether the view contains no bytes.
  bool empty() const {
    return size_ == 0;
  }

 private:
  const uint8_t* data_{nullptr};
  size_t size_{0};
};

/// Provider-owned read-only input. wait() completes host submission. waitOn()
/// additionally orders the first consumer after device production, without a
/// host/device synchronization. Without a ready hook, consume on the stream
/// supplied to the source. A provider installing a ready hook may support a
/// different first-consumer stream and must retain/free storage accordingly.
/// Release only after the last consumer use has been submitted. Subsequent
/// cross-stream uses must join back to that consumer before release; waitOn()
/// is a first-consumer operation, not a general stream-handoff primitive.
/// Streams and memory resources must outlive the batch.
struct DeviceReadBatch {
  /// Retains storage and any source state needed by pending work.
  std::shared_ptr<void> owner;
  /// Contains one view per requested range, preserving input order.
  std::vector<DeviceByteView> data;
  /// Completes host submission; wait() propagates any read failure.
  std::future<void> pending;
  /// Orders device production before the first consumer when necessary.
  std::function<void(cuda::stream_ref)> ready;

  /// Creates an unstarted batch.
  DeviceReadBatch() = default;
  /// Transfers storage and pending work.
  DeviceReadBatch(DeviceReadBatch&&) noexcept = default;
  /// Finishes the previous batch before releasing its storage.
  DeviceReadBatch& operator=(DeviceReadBatch&& other) noexcept;
  /// Finishes pending host work before releasing storage.
  ~DeviceReadBatch();
  /// Distinguishes a started batch from an unstarted one, even without I/O.
  explicit operator bool() const {
    return owner != nullptr;
  }
  /// Completes host submission and propagates its error, if any.
  void wait();
  /// Completes submission and orders production before the consumer stream.
  void waitOn(cuda::stream_ref stream);
};

/// Optional device-input capability, independent of cuDF datasource and types.
/// The source chooses allocation/retention; consumers retain only the batch.
class DeviceReadSource {
 public:
  /// Releases the source interface.
  virtual ~DeviceReadSource() = default;
  /// Starts exact reads, returning immutable device views and retained storage.
  /// Copies range descriptors before returning. The source and memory resources
  /// must outlive submission. The batch must own all referenced storage and
  /// have a non-null owner, including for an empty request. Providers satisfy
  /// the caller's decoder padding requirements and arrange release after queued
  /// consumption on the supplied stream. This capability does not imply
  /// caching.
  virtual DeviceReadBatch readDeviceRangesAsync(
      std::span<const DeviceByteRange> ranges,
      bool serializeSubmissions,
      cuda::stream_ref stream,
      rmm::device_async_resource_ref outputResource,
      rmm::device_async_resource_ref temporaryResource) = 0;
};

} // namespace facebook::velox::gpu
