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
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/buffer/Buffer.h"
#include "velox/common/base/Exceptions.h"
#include "velox/common/memory/MemoryPool.h"
#include "velox/common/process/StackTrace.h"
#include "velox/vector/TypeAliases.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_stream.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <atomic>

namespace facebook::velox::cudf_velox {
namespace {

// Establish stream ordering without a host wait. Destroying a recorded event
// does not cancel the wait already submitted to the destination stream.
cudaError_t joinStreams(cuda::stream_ref source, cuda::stream_ref destination) {
  if (source.get() == destination.get()) {
    return cudaSuccess;
  }
  cudaEvent_t event;
  auto status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaEventRecord(event, source.get());
  if (status == cudaSuccess) {
    status = cudaStreamWaitEvent(destination.get(), event, 0);
  }
  const auto destroyStatus = cudaEventDestroy(event);
  return status == cudaSuccess ? destroyStatus : status;
}

// Reuse a stream per publication thread without putting consumer waits on a
// producer's compute stream. Each payload retains the stream across thread
// exit.
std::shared_ptr<rmm::cuda_stream> reclamationStream() {
  static thread_local int cachedDevice = -1;
  static thread_local std::shared_ptr<rmm::cuda_stream> cachedStream;
  int device;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  if (!cachedStream || device != cachedDevice) {
    cachedStream = std::make_shared<rmm::cuda_stream>();
    cachedDevice = device;
  }
  return cachedStream;
}

void rebindTable(std::unique_ptr<cudf::table>& table, cuda::stream_ref stream) {
  auto columns = table->release();
  for (auto& column : columns) {
    column = cudf::rebind_stream(std::move(*column), stream);
  }
  table = std::make_unique<cudf::table>(std::move(columns));
}

void logDefaultStreamIfNeeded(
    cuda::stream_ref stream,
    const char* constructorName) {
  if (stream != cuda::stream_ref{cudaStream_t{cudaStreamDefault}}) {
    return;
  }
  LOG(WARNING) << constructorName
               << " constructed with default CUDA stream. Backtrace:\n"
               << process::StackTrace().toString();
}

// Returns false once release() has taken the storage.
template <typename TableStorage>
bool hasStorage(const TableStorage& storage) {
  return std::visit(
      [](const auto& table) { return table != nullptr; }, storage);
}

} // namespace

struct CudfVector::SharedStorage {
  // Destroy the table before the stream on which its buffers are freed.
  std::shared_ptr<rmm::cuda_stream> releaseStream{reclamationStream()};
  TableStorage table;
  std::atomic<bool> unsafeToFree{false};

  cuda::stream_ref stream() const {
    return *releaseStream;
  }

  ~SharedStorage() {
    if (unsafeToFree.load()) {
      // A failed CUDA context cannot prove that readers have stopped using
      // storage. Keep the allocations alive rather than aborting in teardown
      // or freeing bytes still referenced by outstanding device work.
      std::visit([](auto& storage) { storage.release(); }, table);
    }
  }
};

CudfVector::~CudfVector() {
  releaseShared();
}

void CudfVector::releaseShared() noexcept {
  if (!sharedStorage_) {
    return;
  }
  const auto status = joinStreams(stream_, sharedStorage_->stream());
  if (status != cudaSuccess) {
    // Destruction cannot throw. Keep buffers alive until the outstanding use
    // completes if event creation/recording fails (e.g. during cancellation).
    LOG(ERROR) << "Cannot order shared device storage release: "
               << cudaGetErrorString(status);
    const auto waitStatus = cudaStreamSynchronize(stream_.get());
    if (waitStatus != cudaSuccess) {
      LOG(ERROR) << "Retaining shared storage after failed CUDA completion: "
                 << cudaGetErrorString(waitStatus);
      sharedStorage_->unsafeToFree.store(true);
    }
  }
  sharedStorage_.reset();
}

CudfVector::CudfVector(
    memory::MemoryPool* pool,
    TypePtr type,
    vector_size_t size,
    std::shared_ptr<SharedStorage> storage,
    cudf::table_view view,
    uint64_t bytes,
    cuda::stream_ref stream)
    : RowVector(pool, std::move(type), nullptr, size, {}),
      sharedStorage_(std::move(storage)),
      tabView_(view),
      stream_(stream),
      flatSize_(bytes) {}

RowVectorPtr CudfVector::shareForRead(memory::MemoryPool* pool) {
  if (!sharedStorage_) {
    auto storage = std::make_shared<SharedStorage>();
    CUDF_CUDA_TRY(joinStreams(stream_, storage->stream()));
    // Publication is exclusive. Bind allocation cleanup once, before readers
    // observe the immutable table. Readiness stays on the producer stream.
    if (auto* table =
            std::get_if<std::unique_ptr<cudf::table>>(&tableStorage_)) {
      VELOX_CHECK_NOT_NULL(*table, "Cannot share released device storage");
      rebindTable(*table, storage->stream());
      tabView_ = (*table)->view();
    } else {
      auto& packed =
          std::get<std::unique_ptr<cudf::packed_table>>(tableStorage_);
      VELOX_CHECK_NOT_NULL(packed, "Cannot share released device storage");
      packed->data.gpu_data->set_stream(storage->stream());
    }
    storage->table = std::move(tableStorage_);
    sharedStorage_ = std::move(storage);
  }
  auto snapshot = std::shared_ptr<CudfVector>(new CudfVector(
      pool, type(), size(), sharedStorage_, tabView_, flatSize_, stream_));
  return snapshot;
}

CudfVector::CudfVector(
    velox::memory::MemoryPool* pool,
    TypePtr type,
    vector_size_t size,
    std::unique_ptr<cudf::table>&& table,
    cuda::stream_ref stream)
    : RowVector(
          pool,
          std::move(type),
          BufferPtr(nullptr),
          size,
          std::vector<VectorPtr>(),
          std::nullopt),
      tableStorage_{std::move(table)},
      stream_{stream} {
  logDefaultStreamIfNeeded(stream_, "CudfVector(table)");
  auto& tablePtr = std::get<std::unique_ptr<cudf::table>>(tableStorage_);
  rebindTable(tablePtr, stream_);
  flatSize_ = tablePtr->alloc_size();
  tabView_ = tablePtr->view();
}

CudfVector::CudfVector(
    velox::memory::MemoryPool* pool,
    TypePtr type,
    vector_size_t size,
    std::unique_ptr<cudf::packed_table>&& packedTable,
    cuda::stream_ref stream)
    : RowVector(
          pool,
          std::move(type),
          BufferPtr(nullptr),
          size,
          std::vector<VectorPtr>(),
          std::nullopt),
      tableStorage_{std::move(packedTable)},
      stream_{stream} {
  logDefaultStreamIfNeeded(stream_, "CudfVector(packed_table)");
  auto& packedPtr =
      std::get<std::unique_ptr<cudf::packed_table>>(tableStorage_);
  packedPtr->data.gpu_data->set_stream(stream_);
  tabView_ = packedPtr->table;
  // For packed table, flatSize is the size of the GPU data buffer
  flatSize_ = packedPtr->data.gpu_data->size();
}

cudf::table_view CudfVector::getTableView() const {
  VELOX_CHECK(
      sharedStorage_ || hasStorage(tableStorage_),
      "CudfVector was already released");
  return tabView_;
}

std::unique_ptr<cudf::table> CudfVector::release() {
  VELOX_CHECK(
      !sharedStorage_, "Shared device storage requires a destination resource");
  const auto* packed =
      std::get_if<std::unique_ptr<cudf::packed_table>>(&tableStorage_);
  return release(
      packed && *packed ? (*packed)->data.gpu_data->memory_resource()
                        : rmm::mr::get_current_device_resource_ref());
}

std::unique_ptr<cudf::table> CudfVector::release(
    rmm::device_async_resource_ref destination) {
  if (sharedStorage_) {
    auto result = std::make_unique<cudf::table>(tabView_, stream_, destination);
    releaseShared();
    tabView_ = cudf::table_view{};
    flatSize_ = 0;
    return result;
  }
  VELOX_CHECK(hasStorage(tableStorage_), "CudfVector was already released");
  if (auto* tablePtr =
          std::get_if<std::unique_ptr<cudf::table>>(&tableStorage_)) {
    // Constructed from owned table - just move it out
    flatSize_ = 0;
    return std::move(*tablePtr);
  }
  // Constructed from packed_table - materialize a table from the view.
  // This copies the data since the view references the packed buffer.
  auto& packedPtr =
      std::get<std::unique_ptr<cudf::packed_table>>(tableStorage_);
  auto materializedTable =
      std::make_unique<cudf::table>(tabView_, stream_, destination);
  stream_.sync();
  // Clear the packed table since we've materialized
  packedPtr.reset();
  flatSize_ = 0;
  return materializedTable;
}

bool CudfVector::rebindStream(cuda::stream_ref stream) {
  if (sharedStorage_) {
    // Only this reader's wrapper changes. Chain its previous use/readiness
    // into the new stream; final release joins that stream to the owner.
    CUDF_CUDA_TRY(joinStreams(stream_, stream));
    stream_ = stream;
    return true;
  }
  if (!hasStorage(tableStorage_)) {
    return false;
  }

  if (auto* tablePtr =
          std::get_if<std::unique_ptr<cudf::table>>(&tableStorage_)) {
    // The constructor and every rebind bind the buffers to 'stream_', so they
    // already free on 'stream'.
    if (stream_.get() == stream.get()) {
      return true;
    }

    rebindTable(*tablePtr, stream);
    tabView_ = (*tablePtr)->view();
    stream_ = stream;
    return true;
  }

  auto& packedPtr =
      std::get<std::unique_ptr<cudf::packed_table>>(tableStorage_);
  packedPtr->data.gpu_data->set_stream(stream);
  stream_ = stream;
  return true;
}

uint64_t CudfVector::estimateFlatSize() const {
  return flatSize_;
}

uint64_t CudfVector::retainedSizeImpl(
    uint64_t& /*totalStringBufferSize*/) const {
  return flatSize_;
}

void orderCudfVectorDeallocationsAfterStream(
    std::span<const CudfVectorPtr> vectors,
    cuda::stream_ref stream) {
  // Rebind every input before checking, so a released one does not leave the
  // rest freeing on their producer streams while 'stream' may still read them.
  bool allRebound = true;
  for (const auto& vector : vectors) {
    VELOX_CHECK_NOT_NULL(vector);
    allRebound &= vector->rebindStream(stream);
  }
  VELOX_CHECK(allRebound, "Cannot rebind a released CudfVector");
}

} // namespace facebook::velox::cudf_velox
