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

#include "velox/vector/LazyVector.h"
#include "velox/common/base/RuntimeMetrics.h"
#include "velox/common/memory/RawVector.h"
#include "velox/common/time/CpuWallTimer.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatMapVector.h"
#include "velox/vector/SelectivityVector.h"

namespace facebook::velox {
namespace {

// Convenience class to record cpu and wall time from construction, updating
// thread local stats at destruction, including input bytes of the vector passed
// as parameter.
class LazyIoStatsRecorder {
 public:
  LazyIoStatsRecorder(VectorPtr* vector) : vector_(vector) {}

  ~LazyIoStatsRecorder() {
    auto cpuDelta = timer_.elapsed();
    addThreadLocalRuntimeStat(
        LazyVector::kWallNanos,
        RuntimeCounter(cpuDelta.wallNanos, RuntimeCounter::Unit::kNanos));
    addThreadLocalRuntimeStat(
        LazyVector::kCpuNanos,
        RuntimeCounter(cpuDelta.cpuNanos, RuntimeCounter::Unit::kNanos));

    if (*vector_) {
      addThreadLocalRuntimeStat(
          LazyVector::kInputBytes,
          RuntimeCounter(
              (*vector_)->estimateFlatSize(), RuntimeCounter::Unit::kBytes));
    }
  }

 private:
  DeltaCpuWallTimeStopWatch timer_;
  VectorPtr* vector_;
};

} // namespace

void VectorLoader::load(
    RowSet rows,
    ValueHook* hook,
    vector_size_t resultSize,
    VectorPtr* result) {
  {
    LazyIoStatsRecorder recorder(result);
    loadInternal(rows, hook, resultSize, result);
  }
  if (hook) {
    // Record number of rows loaded directly into ValueHook bypassing
    // materialization into vector. This counter can be used to understand
    // whether aggregation pushdown is happening or not.
    addThreadLocalRuntimeStat("loadedToValueHook", RuntimeCounter(rows.size()));
  }
}

void VectorLoader::load(
    const SelectivityVector& rows,
    ValueHook* hook,
    vector_size_t resultSize,
    VectorPtr* result,
    memory::MemoryPool* pool) {
  if (rows.isAllSelected()) {
    const auto& indices = DecodedVector::consecutiveIndices();
    VELOX_DCHECK(!indices.empty());
    if (rows.end() <= indices.size()) {
      load(
          RowSet(&indices[rows.begin()], rows.end() - rows.begin()),
          hook,
          resultSize,
          result);
      return;
    }
  }
  raw_vector<vector_size_t> positions(rows.countSelected(), pool);
  simd::indicesOfSetBits(
      rows.allBits(), rows.begin(), rows.end(), positions.data());
  load(positions, hook, resultSize, result);
}

VectorPtr LazyVector::slice(vector_size_t offset, vector_size_t length) const {
  VELOX_CHECK(isLoaded(), "Cannot take slice on unloaded lazy vector");
  VELOX_DCHECK(vector_);
  return vector_->slice(offset, length);
}

//   static
// This gurantee that vectors are loaded but it does not gurantee that vector
// will be free of any nested lazy vector i.e, references to Loaded lazy vectors
// are not replaced with their underlying loaded vector.
void LazyVector::ensureLoadedRows(
    const VectorPtr& vector,
    const SelectivityVector& rows) {
  if (isLazyNotLoaded(*vector)) {
    DecodedVector decoded;
    SelectivityVector baseRows;
    ensureLoadedRows(vector, rows, decoded, baseRows);
  } else {
    // Even if LazyVectors have been loaded, via some other path, this
    // is needed for initializing wrappers.
    vector->loadedVector();
  }
}

namespace {
// Inspect metadata and initialize fully loaded descendants bottom-up. Unlike
// isLazyNotLoaded(), this includes lazy values inside all containers. This
// never loads unresolved values or allocates row/element selection bitmaps.
bool initializeIfFullyLoaded(const VectorPtr& vector) {
  if (!vector) {
    return true;
  }
  switch (vector->encoding()) {
    case VectorEncoding::Simple::LAZY: {
      const auto* lazy = vector->as<LazyVector>();
      if (!lazy->isLoaded() ||
          !initializeIfFullyLoaded(lazy->loadedVectorShared())) {
        return false;
      }
      break;
    }
    case VectorEncoding::Simple::DICTIONARY:
    case VectorEncoding::Simple::CONSTANT:
    case VectorEncoding::Simple::SEQUENCE:
      if (!initializeIfFullyLoaded(vector->valueVector())) {
        return false;
      }
      break;
    case VectorEncoding::Simple::ROW:
      for (const auto& child : vector->as<RowVector>()->children()) {
        if (!initializeIfFullyLoaded(child)) {
          return false;
        }
      }
      break;
    case VectorEncoding::Simple::ARRAY:
      if (!initializeIfFullyLoaded(vector->as<ArrayVector>()->elements())) {
        return false;
      }
      break;
    case VectorEncoding::Simple::MAP: {
      const auto* map = vector->as<MapVector>();
      if (!initializeIfFullyLoaded(map->mapKeys()) ||
          !initializeIfFullyLoaded(map->mapValues())) {
        return false;
      }
      break;
    }
    case VectorEncoding::Simple::FLAT_MAP: {
      const auto* map = vector->as<FlatMapVector>();
      if (!initializeIfFullyLoaded(map->distinctKeys())) {
        return false;
      }
      for (const auto& values : map->mapValues()) {
        if (!initializeIfFullyLoaded(values)) {
          return false;
        }
      }
      break;
    }
    default:
      break;
  }
  vector->loadedVector();
  return true;
}

// Given a SelectivityVector 'rows', updates 'baseRows' selecting the rows
// in the base vector that should be loaded.
void selectBaseRowsToLoad(
    const DecodedVector& decoded,
    SelectivityVector& baseRows,
    const SelectivityVector& rows) {
  VELOX_DCHECK(
      decoded.base()->encoding() == VectorEncoding::Simple::ROW ||
      decoded.base()->encoding() == VectorEncoding::Simple::FLAT_MAP ||
      decoded.base()->encoding() == VectorEncoding::Simple::ARRAY ||
      decoded.base()->encoding() == VectorEncoding::Simple::MAP ||
      decoded.base()->encoding() == VectorEncoding::Simple::LAZY);

  if (!rows.hasSelections()) {
    baseRows.resizeFill(0, false);
    return;
  }

  auto deselectNullsIdentity = [&]() {
    if (decoded.base()->rawNulls()) {
      baseRows.deselectNulls(
          decoded.base()->rawNulls(), rows.begin(), rows.end());
    }
  };

  if (decoded.isIdentityMapping() && rows.isAllSelected()) {
    baseRows.resizeFill(rows.end(), true);
    deselectNullsIdentity();
    return;
  }

  if (decoded.isIdentityMapping()) {
    baseRows.resizeFill(rows.end(), false);
    baseRows.select(rows);
    deselectNullsIdentity();
  } else if (decoded.isConstantMapping()) {
    baseRows.resizeFill(decoded.index(0) + 1, false);
    if (!decoded.isNullAt(rows.begin())) {
      baseRows.setValid(decoded.index(0), true);
    }
  } else {
    baseRows.resizeFill(decoded.base()->size(), false);
    rows.applyToSelected([&](vector_size_t row) {
      if (!decoded.isNullAt(row)) {
        baseRows.setValid(decoded.index(row), true);
      }
    });
  }

  baseRows.updateBounds();
}
} // namespace

// static
void LazyVector::ensureLoadedRowsImpl(
    const VectorPtr& vector,
    DecodedVector& decoded,
    const SelectivityVector& rows,
    SelectivityVector& baseRows,
    bool loadAllChildren) {
  if (decoded.base()->encoding() != VectorEncoding::Simple::LAZY) {
    if (decoded.base()->encoding() == VectorEncoding::Simple::ROW &&
        (loadAllChildren || isLazyNotLoaded(*decoded.base()))) {
      auto* rowVector = decoded.base()->asUnchecked<RowVector>();
      selectBaseRowsToLoad(decoded, baseRows, rows);
      DecodedVector decodedChild;
      SelectivityVector childRows;
      for (const auto& child : rowVector->children()) {
        // Selective readers leave unprojected fields absent.
        if (!child) {
          continue;
        }
        if (loadAllChildren) {
          ensureLoadedRowsRecursively(child, baseRows);
          continue;
        }
        decodedChild.decode(*child, baseRows, false);
        ensureLoadedRowsImpl(
            child, decodedChild, baseRows, childRows, loadAllChildren);
      }
      rowVector->invalidateContainsLazyNotLoaded();
      vector->loadedVector();
    } else if (
        loadAllChildren &&
        decoded.base()->encoding() == VectorEncoding::Simple::FLAT_MAP) {
      const auto* map = decoded.base()->as<FlatMapVector>();
      selectBaseRowsToLoad(decoded, baseRows, rows);
      // Keys describe the channels independently of selected map rows.
      ensureLoadedRowsRecursively(
          map->distinctKeys(), SelectivityVector(map->numDistinctKeys()));
      if (map->rawNulls()) {
        baseRows.deselectNulls(
            map->rawNulls(), baseRows.begin(), baseRows.end());
      }
      SelectivityVector valueRows;
      for (column_index_t channel = 0; channel < map->numDistinctKeys();
           ++channel) {
        const auto& values = map->mapValuesAt(channel);
        if (!values) {
          continue;
        }
        valueRows = baseRows;
        if (const auto* inMap = map->rawInMapsAt(channel)) {
          valueRows.deselectNulls(inMap, baseRows.begin(), baseRows.end());
        }
        ensureLoadedRowsRecursively(values, valueRows);
      }
      vector->loadedVector();
    } else if (
        loadAllChildren &&
        (decoded.base()->encoding() == VectorEncoding::Simple::ARRAY ||
         decoded.base()->encoding() == VectorEncoding::Simple::MAP)) {
      // A container may hold nested FlatMaps whose value channels are lazy.
      const auto* container = decoded.base()->as<ArrayVectorBase>();
      selectBaseRowsToLoad(decoded, baseRows, rows);
      const bool isMap = container->encoding() == VectorEncoding::Simple::MAP;
      const auto& values = isMap ? container->as<MapVector>()->mapValues()
                                 : container->as<ArrayVector>()->elements();
      SelectivityVector elements(values->size(), false);
      baseRows.applyToSelected([&](auto row) {
        if (!container->isNullAt(row)) {
          const auto offset = container->offsetAt(row);
          elements.setValidRange(offset, offset + container->sizeAt(row), true);
        }
      });
      elements.updateBounds();
      if (isMap) {
        ensureLoadedRowsRecursively(
            container->as<MapVector>()->mapKeys(), elements);
      }
      ensureLoadedRowsRecursively(values, elements);
      vector->loadedVector();
    }
    return;
  }
  // Base vector is lazy.
  auto baseLazyVector = decoded.base()->asUnchecked<LazyVector>();

  if (!baseLazyVector->isLoaded()) {
    // Create rowSet.
    raw_vector<vector_size_t> rowNumbers(baseLazyVector->pool());
    RowSet rowSet;
    if (decoded.isConstantMapping()) {
      if (rows.hasSelections()) {
        rowNumbers.push_back(decoded.index(rows.begin()));
      }
      rowSet = RowSet(rowNumbers);
    } else if (decoded.isIdentityMapping()) {
      if (rows.isAllSelected()) {
        auto iota = velox::iota(rows.end(), rowNumbers);
        rowSet = RowSet(iota, rows.end());
      } else {
        rowNumbers.resize(rows.end());
        rowNumbers.resize(
            simd::indicesOfSetBits(
                rows.asRange().bits(), 0, rows.end(), rowNumbers.data()));
        rowSet = RowSet(rowNumbers);
      }
    } else {
      selectBaseRowsToLoad(decoded, baseRows, rows);
      rowNumbers.resize(baseRows.end());
      rowNumbers.resize(
          simd::indicesOfSetBits(
              baseRows.asRange().bits(), 0, baseRows.end(), rowNumbers.data()));

      rowSet = RowSet(rowNumbers);
    }

    baseLazyVector->load(rowSet, nullptr);
  }

  // The loaded vector can itself also be lazy, so we load recursively.
  if (loadAllChildren || isLazyNotLoaded(*baseLazyVector->vector_)) {
    // Preserve the original selection when descending into a loaded result.
    selectBaseRowsToLoad(decoded, baseRows, rows);
    if (loadAllChildren) {
      ensureLoadedRowsRecursively(baseLazyVector->vector_, baseRows);
    } else {
      decoded.decode(*baseLazyVector->vector_, baseRows, false);
      SelectivityVector nestedRows;
      ensureLoadedRowsImpl(
          baseLazyVector->vector_, decoded, baseRows, nestedRows);
    }
  }

  // An explicit call to loadedVector() is necessary to allow for proper
  // initialization of dictionaries, sequences, etc. on top of lazy vector
  // after it has been loaded, even if loaded via some other path.
  vector->loadedVector();
}

// static
void LazyVector::ensureLoadedRows(
    const VectorPtr& vector,
    const SelectivityVector& rows,
    DecodedVector& decoded,
    SelectivityVector& baseRows) {
  decoded.decode(*vector, rows, false);
  ensureLoadedRowsImpl(vector, decoded, rows, baseRows);
}

void LazyVector::ensureLoadedRowsRecursively(
    const VectorPtr& vector,
    const SelectivityVector& rows) {
  if (initializeIfFullyLoaded(vector)) {
    // No decoding or selections for trees that have no work left to load.
    return;
  }
  // Peel wrappers one level at a time so sequences, including ones underneath
  // dictionaries/constants, never reach DecodedVector (which rejects them).
  const auto encoding = vector->encoding();
  if (encoding == VectorEncoding::Simple::DICTIONARY ||
      encoding == VectorEncoding::Simple::CONSTANT ||
      encoding == VectorEncoding::Simple::SEQUENCE) {
    const auto& values = vector->valueVector();
    SelectivityVector selected(values->size(), false);
    const auto* info = vector->wrapInfo()->as<vector_size_t>();
    if (encoding == VectorEncoding::Simple::CONSTANT) {
      if (rows.hasSelections()) {
        selected.setValid(info[0], true);
      }
    } else if (encoding == VectorEncoding::Simple::DICTIONARY) {
      rows.applyToSelected([&](auto row) {
        if (!vector->rawNulls() || !bits::isBitNull(vector->rawNulls(), row)) {
          selected.setValid(info[row], true);
        }
      });
    } else {
      vector_size_t run = 0;
      vector_size_t end = values->size() ? info[0] : 0;
      rows.applyToSelected([&](auto row) {
        while (row >= end) {
          end += info[++run];
        }
        selected.setValid(run, true);
      });
    }
    selected.updateBounds();
    ensureLoadedRowsRecursively(values, selected);
    vector->loadedVector();
    return;
  }
  DecodedVector decoded;
  SelectivityVector baseRows;
  decoded.decode(*vector, rows, false);
  ensureLoadedRowsImpl(vector, decoded, rows, baseRows, true);
  // Also initialize wrappers over independently loaded lazy bases.
  vector->loadedVector();
}

void LazyVector::validate(const VectorValidateOptions& options) const {
  if (isLoaded() || options.loadLazy) {
    auto loadedVector = this->loadedVector();
    VELOX_CHECK_NOT_NULL(loadedVector);
    loadedVector->validate(options);
  }
}

void LazyVector::load(RowSet rows, ValueHook* hook) const {
  VELOX_CHECK(!allLoaded_, "A LazyVector can be loaded at most once");
  VELOX_CHECK(!hook || supportsHook());

  allLoaded_ = true;
  if (rows.empty()) {
    vector_ = BaseVector::createNullConstant(type_, size(), pool_);
    return;
  }

  // The loader expect structs to be pre-allocated.
  if (!vector_ && type_->kind() == TypeKind::ROW) {
    vector_ = BaseVector::create(type_, rows.back() + 1, pool_);
  }

  // We do not call prepareForReuse on vector_ in purpose, the loader handles
  // that.
  loader_->load(rows, hook, size(), &vector_);
  if (!hook) {
    VELOX_CHECK_GE(vector_->size(), size());
  }
}

void LazyVector::loadVectorInternal() const {
  if (!allLoaded_) {
    if (!vector_) {
      vector_ = BaseVector::create(type_, 0, pool_);
    }
    SelectivityVector allRows(BaseVector::length_);
    loader_->load(allRows, nullptr, size(), &vector_, pool_);
    VELOX_CHECK_NOT_NULL(vector_);
    if (vector_->encoding() == VectorEncoding::Simple::LAZY) {
      vector_ = vector_->asUnchecked<LazyVector>()->loadedVectorShared();
    } else {
      // If the load produced a wrapper, load the wrapped vector.
      vector_->loadedVector();
    }
    allLoaded_ = true;
    const_cast<LazyVector*>(this)->BaseVector::nulls_ = vector_->nulls_;
    if (BaseVector::nulls_) {
      const_cast<LazyVector*>(this)->BaseVector::rawNulls_ =
          BaseVector::nulls_->as<uint64_t>();
    }
  } else {
    VELOX_CHECK_NOT_NULL(vector_);
  }
}
} // namespace facebook::velox
