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

#include "velox/common/memory/MemoryPool.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/SharedVector.h"
#include "velox/vector/TypeAliases.h"

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <memory>
#include <span>
#include <utility>
#include <variant>

namespace facebook::velox::cudf_velox {

// Vector class which holds GPU data from cuDF.
// Can be constructed either from an owned cudf::table or from packed_table.
// When constructed from packed_table, the data remains packed and tabView_
// references the table view inside packed_table without copying the underlying
// GPU data.
//
// CudfVector assumes that the input table data is already ready to consume.
// Constructors and unshared rebindStream() do not insert stream waits for work
// that populated the input buffers; callers must establish that ordering before
// constructing or rebinding a CudfVector.
class CudfVector : public RowVector, public SharedVector {
 public:
  /// Constructs a CudfVector from an owned cudf::table.
  /// The table's buffers are freed on 'stream', or on the stream passed to a
  /// later rebindStream().
  CudfVector(
      velox::memory::MemoryPool* pool,
      TypePtr type,
      vector_size_t size,
      std::unique_ptr<cudf::table>&& table,
      cuda::stream_ref stream);

  /// Constructs a CudfVector from packed_table.
  /// The packed data is retained and tabView_ references the table view inside
  /// packed_table. This avoids copying the underlying GPU data.
  /// The packed buffer is freed on 'stream', or on the stream passed to a later
  /// rebindStream().
  CudfVector(
      velox::memory::MemoryPool* pool,
      TypePtr type,
      vector_size_t size,
      std::unique_ptr<cudf::packed_table>&& packedTable,
      cuda::stream_ref stream);

  ~CudfVector() override;

  /// Shares immutable device buffers with independent consumer metadata.
  /// Final deallocation uses a separate stream, retained by the shared payload.
  RowVectorPtr shareForRead(memory::MemoryPool* pool) override;

  /// Build an immutable output from shared input columns and newly owned
  /// columns without copying passthroughs. Both vectors have output width;
  /// each null owned column selects inputChannels[i] from input. Retains the
  /// complete shared input allocation, including columns not selected.
  static std::shared_ptr<CudfVector> withSharedColumns(
      const std::shared_ptr<CudfVector>& input,
      TypePtr outputType,
      std::vector<std::unique_ptr<cudf::column>> columns,
      const std::vector<column_index_t>& inputChannels);

  bool isShared() const {
    return sharedStorage_ != nullptr;
  }

  cuda::stream_ref stream() const {
    return stream_;
  }

  /// Returns a view of the underlying table. Throws after release().
  cudf::table_view getTableView() const;

  /// Releases ownership of the underlying table. Shared storage is copied on
  /// the consumer stream so other readers retain their original buffers.
  /// If constructed from packed_table, materializes a table from the view
  /// first (which copies the data). Shared storage requires the explicit
  /// destination overload. Throws if called more than once.
  std::unique_ptr<cudf::table> release();

  /// Uses 'destination' for copies required to obtain exclusive ownership.
  /// Already exclusive table storage is transferred without allocating.
  std::unique_ptr<cudf::table> release(
      rmm::device_async_resource_ref destination);

  /// Rebinds owned table buffers to use 'stream' for future deallocation.
  /// Unshared storage only changes its deallocation association; the caller
  /// establishes readiness. Shared snapshots also order this reader's previous
  /// stream before the new stream without mutating the other readers.
  /// Returns false if the storage was already taken by release().
  bool rebindStream(cuda::stream_ref stream);

  uint64_t estimateFlatSize() const override;

 private:
  // Owns an immutable device table and its final deallocation stream.
  struct SharedStorage;

  // Constructs a private view of an already published payload.
  CudfVector(
      memory::MemoryPool* pool,
      TypePtr type,
      vector_size_t size,
      std::shared_ptr<SharedStorage> storage,
      cudf::table_view view,
      uint64_t bytes,
      cuda::stream_ref stream);

  // Joins this consumer into the shared owner before dropping its reference.
  void releaseShared() noexcept;

  uint64_t retainedSizeImpl(uint64_t& totalStringBufferSize) const override;

  // Storage for either an owned table or packed table.
  // Only one is active at a time - using variant enforces this at compile time.
  using TableStorage = std::variant<
      std::unique_ptr<cudf::table>,
      std::unique_ptr<cudf::packed_table>>;
  TableStorage tableStorage_;

  // Set only after publication; owned buffers then live in SharedStorage.
  std::shared_ptr<SharedStorage> sharedStorage_;

  // Views owned, packed, or shared storage. Invalid after release().
  cudf::table_view tabView_;

  cuda::stream_ref stream_;
  uint64_t flatSize_;
};

using CudfVectorPtr = std::shared_ptr<CudfVector>;

/// Orders CudfVector deallocations after work on a target stream.
///
/// Rebinds each input's device buffers to 'stream' so stream-ordered memory
/// resources free the inputs after prior work on that stream. Throws if any
/// input was passed to CudfVector::release(), after rebinding the others.
void orderCudfVectorDeallocationsAfterStream(
    std::span<const CudfVectorPtr> vectors,
    cuda::stream_ref stream);

} // namespace facebook::velox::cudf_velox
