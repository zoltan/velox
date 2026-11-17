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
#include "velox/exec/SharedResultBuffer.h"
#include "velox/vector/SharedVector.h"

#include "velox/vector/ConstantVector.h"
#include "velox/vector/DictionaryVector.h"
#include "velox/vector/FlatMapVector.h"
#include "velox/vector/LazyVector.h"
#include "velox/vector/SequenceVector.h"
#include "velox/vector/VectorTypeUtils.h"

namespace facebook::velox::exec::detail {

namespace {

// Clone only mutable container/wrapper metadata. Scalar leaves and all data
// buffers are retained by reference. Ordinary Velox writable/reuse APIs detach
// shared data if a producer or consumer actually needs to modify it.
VectorPtr shareVector(
    const VectorPtr& vector,
    memory::MemoryPool* pool,
    bool publication = false);

template <TypeKind kind>
VectorPtr shareWrapper(
    const VectorPtr& vector,
    memory::MemoryPool* pool,
    bool publication) {
  using T = typename KindToFlatVector<kind>::WrapperType;
  // Publication already loaded the selected rows. A wrapper can still need
  // initialization if its lazy base was loaded independently.
  vector->loadedVector();
  auto values = shareVector(vector->valueVector(), pool, publication);
  if (vector->encoding() == VectorEncoding::Simple::DICTIONARY) {
    return std::make_shared<DictionaryVector<T>>(
        pool,
        vector->nulls(),
        vector->size(),
        std::move(values),
        vector->wrapInfo());
  }
  // SequenceVector has a mutable lookup cursor even for read-only access.
  return std::make_shared<SequenceVector<T>>(
      pool, vector->size(), std::move(values), vector->wrapInfo());
}

VectorPtr shareVector(
    const VectorPtr& vector,
    memory::MemoryPool* pool,
    bool publication) {
  if (!vector) {
    return nullptr;
  }
  if (auto* shared = dynamic_cast<SharedVector*>(vector.get())) {
    return shared->shareForRead(pool);
  }
  const auto size = vector->size();
  switch (vector->encoding()) {
    case VectorEncoding::Simple::LAZY:
      // Resolve reader-backed lazy data before the producer advances.
      return shareVector(
          BaseVector::loadedVectorShared(vector), pool, publication);
    case VectorEncoding::Simple::ROW: {
      std::vector<VectorPtr> children;
      children.reserve(vector->as<RowVector>()->children().size());
      for (const auto& child : vector->as<RowVector>()->children()) {
        children.push_back(shareVector(child, pool, publication));
      }
      auto result = std::make_shared<RowVector>(
          pool, vector->type(), vector->nulls(), size, std::move(children));
      // Initialize this new snapshot while it is still private to the producer.
      if (publication) {
        result->loadedVector();
      }
      return result;
    }
    case VectorEncoding::Simple::ARRAY: {
      const auto* array = vector->as<ArrayVector>();
      return std::make_shared<ArrayVector>(
          pool,
          vector->type(),
          vector->nulls(),
          size,
          array->offsets(),
          array->sizes(),
          shareVector(array->elements(), pool, publication));
    }
    case VectorEncoding::Simple::MAP: {
      const auto* map = vector->as<MapVector>();
      return std::make_shared<MapVector>(
          pool,
          vector->type(),
          vector->nulls(),
          size,
          map->offsets(),
          map->sizes(),
          shareVector(map->mapKeys(), pool, publication),
          shareVector(map->mapValues(), pool, publication));
    }
    case VectorEncoding::Simple::FLAT_MAP: {
      const auto* map = vector->as<FlatMapVector>();
      std::vector<VectorPtr> values;
      values.reserve(map->mapValues().size());
      for (const auto& value : map->mapValues()) {
        values.push_back(shareVector(value, pool, publication));
      }
      auto result = std::make_shared<FlatMapVector>(
          pool,
          vector->type(),
          vector->nulls(),
          size,
          shareVector(map->distinctKeys(), pool, publication),
          std::move(values),
          map->inMaps(),
          std::nullopt,
          map->hasSortedKeys());
      result->shareKeyIndex(*map);
      return result;
    }
    case VectorEncoding::Simple::DICTIONARY:
    case VectorEncoding::Simple::SEQUENCE:
      return VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
          shareWrapper, vector->typeKind(), vector, pool, publication);
    case VectorEncoding::Simple::CONSTANT:
      vector->loadedVector();
      if (vector->mayHaveNulls() ||
          (vector->typeKind() != TypeKind::ROW &&
           vector->typeKind() != TypeKind::ARRAY &&
           vector->typeKind() != TypeKind::MAP)) {
        // Null/scalar constant values (including long strings) are immutable.
        // Do not use slice(), which copies the payload of inline string
        // constants.
        return vector;
      }
      return std::make_shared<ConstantVector<ComplexType>>(
          pool,
          size,
          vector->as<ConstantVector<ComplexType>>()->index(),
          shareVector(vector->valueVector(), pool, publication));
    case VectorEncoding::Simple::FLAT:
    case VectorEncoding::Simple::BIASED:
      return vector;
    default:
      VELOX_UNSUPPORTED("Cannot share vector encoding {}", vector->encoding());
  }
}

} // namespace

SharedResultBuffer::SharedResultBuffer(
    std::vector<core::PlanNodeId> readers,
    CteOptions options)
    : options_(options) {
  VELOX_CHECK_GT(readers.size(), 0);
  VELOX_CHECK(!options.maxRetainedBytes || *options.maxRetainedBytes > 0);
  readers_.reserve(readers.size());
  for (const auto& id : readers) {
    VELOX_CHECK(readerIndices_.emplace(id, readers_.size()).second);
    readers_.emplace_back();
  }
}

uint32_t SharedResultBuffer::readerIndex(const core::PlanNodeId& id) const {
  return readerIndices_.at(id);
}

void SharedResultBuffer::setProducerCount(uint32_t count) {
  std::lock_guard<std::mutex> lock(mutex_);
  VELOX_CHECK_GT(count, 0);
  VELOX_CHECK(numProducers_ == 0 || count == numProducers_);
  numProducers_ = count;
}

void SharedResultBuffer::setReaderDriverCount(uint32_t reader, uint32_t count) {
  std::lock_guard<std::mutex> lock(mutex_);
  VELOX_CHECK_GT(count, 0);
  auto& state = readers_.at(reader);
  VELOX_CHECK(state.numDrivers == 0 || count == state.numDrivers);
  state.numDrivers = count;
}

uint64_t SharedResultBuffer::batchMetadataBytes(const RowVector& batch) {
  return sizeof(RowVector) + sizeof(Batch) + 2 * sizeof(void*) +
      batch.children().capacity() * sizeof(VectorPtr);
}

uint64_t SharedResultBuffer::batchBytes(const RowVector& batch) {
  return batch.retainedSize() + batchMetadataBytes(batch);
}

SharedResultBuffer::Batch::Batch(
    RowVectorPtr data,
    uint64_t bytes,
    uint64_t metadataBytes,
    memory::MemoryPool* pool)
    : data(std::move(data)),
      bytes(bytes),
      metadataBytes(metadataBytes),
      metadataPool(pool->shared_from_this()) {
  metadataPool->reportExternalAllocation(metadataBytes);
}

SharedResultBuffer::Batch::Batch(Batch&& other) noexcept
    : data(std::move(other.data)),
      bytes(other.bytes),
      metadataBytes(other.metadataBytes),
      metadataPool(std::move(other.metadataPool)) {}

SharedResultBuffer::Batch::~Batch() {
  // Both arbitrary payload releasers and pool releases run outside the lock.
  data.reset();
  if (metadataPool) {
    metadataPool->reportExternalFree(metadataBytes);
  }
}

void SharedResultBuffer::waitLocked(ContinueFuture* future, Waiters& waiters) {
  waiters.emplace_back("SharedResultBuffer");
  *future = waiters.back().getSemiFuture();
}

void SharedResultBuffer::takeWaitersLocked(
    Waiters& waiters,
    std::vector<ContinuePromise>& ready,
    bool all) {
  while (!waiters.empty()) {
    ready.push_back(std::move(waiters.front()));
    waiters.pop_front();
    if (!all) {
      break;
    }
  }
}

bool SharedResultBuffer::allReadersClosedLocked() const {
  return std::all_of(readers_.begin(), readers_.end(), [](const auto& reader) {
    return reader.closed();
  });
}

bool SharedResultBuffer::allReadersClosed() {
  std::lock_guard<std::mutex> lock(mutex_);
  return allReadersClosedLocked();
}

void SharedResultBuffer::add(
    RowVectorPtr batch,
    memory::MemoryPool* metadataPool) {
  if (!metadataPool) {
    metadataPool = batch->pool();
  }
  // Resolve only logically visible rows before the producer advances. This
  // preserves dictionary/constant selections and parent ROW null masks before
  // snapshotting recursively visits the underlying vectors.
  if (!dynamic_cast<SharedVector*>(batch.get())) {
    LazyVector::ensureLoadedRowsRecursively(
        batch, SelectivityVector(batch->size()));
  }
  batch = std::static_pointer_cast<RowVector>(
      shareVector(batch, batch->pool(), true));
  const auto bytes = batchBytes(*batch);
  const auto metadataBytes = batchMetadataBytes(*batch);
  Batch entry{std::move(batch), bytes, metadataBytes, metadataPool};
  std::vector<ContinuePromise> ready;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    VELOX_CHECK(!cancelled_, "Shared result cancelled");
    VELOX_CHECK_GT(numProducers_, 0);
    VELOX_CHECK(!finishedLocked());
    if (allReadersClosedLocked()) {
      return;
    }
    if (options_.maxRetainedBytes) {
      VELOX_USER_CHECK_LE(
          bytes,
          *options_.maxRetainedBytes - bytes_,
          "Shared result retained-byte limit exceeded ({} bytes); "
          "increase maxRetainedBytes or avoid sharing this subplan",
          *options_.maxRetainedBytes);
    }
    // Keep entry outside the lock scope so an allocation failure during
    // insertion cannot destroy its payload while holding the mutex.
    batches_.push_back(std::move(entry));
    bytes_ += bytes;
    peakBytes_ = std::max(peakBytes_, bytes_);
    // One new batch per logical reader: wake one waiting driver in each group.
    for (auto& reader : readers_) {
      if (!reader.closed()) {
        takeWaitersLocked(reader.promises, ready, false);
      }
    }
  }
  notify(std::move(ready));
}

uint64_t SharedResultBuffer::finish() {
  std::vector<ContinuePromise> ready;
  uint64_t peakBytes = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    VELOX_CHECK_LT(finishedProducers_, numProducers_);
    ++finishedProducers_;
    if (finishedLocked()) {
      peakBytes = peakBytes_;
      for (auto& reader : readers_) {
        takeWaitersLocked(reader.promises, ready, true);
      }
    }
  }
  notify(std::move(ready));
  return peakBytes;
}

bool SharedResultBuffer::readerBlocked(
    uint32_t reader,
    ContinueFuture* future) {
  std::lock_guard<std::mutex> lock(mutex_);
  VELOX_CHECK(!cancelled_, "Shared result cancelled");
  auto& state = readers_.at(reader);
  VELOX_CHECK_GT(state.numDrivers, 0);
  if (state.closed() || state.next < endLocked() || finishedLocked()) {
    return false;
  }
  waitLocked(future, state.promises);
  return true;
}

void SharedResultBuffer::releaseLocked(ReleasedBatches& released) {
  uint64_t next = endLocked();
  for (const auto& reader : readers_) {
    if (!reader.closed()) {
      next = std::min(next, reader.next);
    }
  }
  while (first_ < next) {
    released.push_back(std::move(batches_.front()));
    bytes_ -= released.back().bytes;
    batches_.pop_front();
    ++first_;
  }
}

RowVectorPtr SharedResultBuffer::next(
    uint32_t reader,
    memory::MemoryPool* pool) {
  RowVectorPtr batch;
  ReleasedBatches released;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    VELOX_CHECK(!cancelled_, "Shared result cancelled");
    auto& state = readers_.at(reader);
    VELOX_CHECK_GT(state.numDrivers, 0);
    if (state.closed() || state.next == endLocked()) {
      return nullptr;
    }
    const auto& entry = batches_[state.next - first_];
    batch = entry.data;
    // Claim before unlocking. The local pointer pins the snapshot until the
    // consumer obtains its own references, even if the queue releases it.
    ++state.next;
    releaseLocked(released);
  }
  // Only metadata is constructed; payload buffers remain shared. Downstream
  // references keep them alive after the queue releases its reference.
  return std::static_pointer_cast<RowVector>(shareVector(batch, pool));
}

bool SharedResultBuffer::atEnd(uint32_t reader) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto& state = readers_.at(reader);
  VELOX_CHECK_GT(state.numDrivers, 0);
  return state.closed() || (finishedLocked() && state.next == endLocked());
}

void SharedResultBuffer::closeReader(uint32_t reader) {
  std::vector<ContinuePromise> ready;
  ReleasedBatches released;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& state = readers_.at(reader);
    VELOX_CHECK_LT(state.closedDrivers, state.numDrivers);
    ++state.closedDrivers;
    if (state.closed()) {
      takeWaitersLocked(state.promises, ready, true);
      releaseLocked(released);
    } else if (state.next < endLocked()) {
      // A notified driver may close without claiming its batch. Hand the
      // available work to a peer rather than stranding it until EOF.
      takeWaitersLocked(state.promises, ready, false);
    }
  }
  notify(std::move(ready));
}

void SharedResultBuffer::cancel() {
  std::vector<ContinuePromise> ready;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    cancelled_ = true;
    for (auto& reader : readers_) {
      takeWaitersLocked(reader.promises, ready, true);
    }
  }
  notify(std::move(ready));
}
} // namespace facebook::velox::exec::detail
