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

#include <folly/small_vector.h>
#include <atomic>
#include <deque>
#include "velox/exec/Cte.h"
#include "velox/exec/JoinBridge.h"

namespace facebook::velox::exec::detail {

/// Task-owned multicast queue. Each logical reader has one cursor and its
/// drivers claim disjoint batches. Vector metadata is private to each consumer;
/// immutable payload buffers are shared using their normal reference counts.
class SharedResultBuffer final : public JoinBridge {
 public:
  SharedResultBuffer(std::vector<core::PlanNodeId> readers, CteOptions options);

  uint32_t readerIndex(const core::PlanNodeId& id) const;

  // Planned counts must be configured before the corresponding driver does
  // any work. Repeated calls validate the count without resetting progress.
  void setProducerCount(uint32_t count);
  void setReaderDriverCount(uint32_t reader, uint32_t count);

  // Vector storage plus a nonzero estimate of the batch object's/queue's
  // overhead. In particular zero-column batches cannot bypass byte limits.
  static uint64_t batchBytes(const RowVector& batch);

  // Retains payload buffers without copying, snapshots container metadata, and
  // resolves lazy inputs before publication. The retention limit is enforced
  // before publishing each batch; producers never wait on readers.
  // Queue metadata is charged to metadataPool (normally the sink operator's
  // pool); defaults to the batch pool for standalone callers. Reservations
  // can arbitrate and therefore happen outside the queue lock.
  void add(RowVectorPtr batch, memory::MemoryPool* metadataPool = nullptr);
  // Once per producer driver; EOF becomes visible after all producers finish.
  // Returns peak retained bytes for the final producer, zero for earlier ones.
  uint64_t finish();
  bool readerBlocked(uint32_t reader, ContinueFuture* future);
  RowVectorPtr next(uint32_t reader, memory::MemoryPool* pool);
  bool atEnd(uint32_t reader);
  // Once per reader driver; only the last close retires the logical reader.
  void closeReader(uint32_t reader);
  bool allReadersClosed();
  void cancel() override;

 private:
  struct Reader {
    Reader() = default;
    Reader(const Reader&) = delete;

    bool closed() const {
      return numDrivers != 0 && closedDrivers == numDrivers;
    }

    // Polling snapshots are published under mutex_. Claiming and waiter
    // registration still recheck the authoritative state under that mutex.
    std::atomic<uint64_t> nextSnapshot{0};
    std::atomic<bool> closedSnapshot{false};
    std::atomic<bool> configured{false};
    uint64_t next{0};
    uint32_t numDrivers{0};
    uint32_t closedDrivers{0};
    std::deque<ContinuePromise> promises;
  };
  struct Batch {
    Batch(
        RowVectorPtr data,
        uint64_t bytes,
        uint64_t metadataBytes,
        memory::MemoryPool* pool);
    Batch(Batch&& other) noexcept;
    ~Batch();

    RowVectorPtr data;
    uint64_t bytes;
    uint64_t metadataBytes;
    std::shared_ptr<memory::MemoryPool> metadataPool;
  };
  static uint64_t batchMetadataBytes(const RowVector& batch);
  using Waiters = std::deque<ContinuePromise>;
  static void waitLocked(ContinueFuture* future, Waiters& waiters);
  static void takeWaitersLocked(
      Waiters& waiters,
      std::vector<ContinuePromise>& ready,
      bool all);
  // The normal claim releases at most one batch; avoid a heap allocation.
  // Callers own this list outside the lock scope so buffer releasers and opaque
  // value destructors can safely reenter the queue.
  using ReleasedBatches = folly::small_vector<Batch, 1>;
  void releaseLocked(ReleasedBatches& released);
  bool allReadersClosedLocked() const;
  uint64_t endLocked() const {
    return first_ + batches_.size();
  }
  bool finishedLocked() const {
    return numProducers_ != 0 && finishedProducers_ == numProducers_;
  }

  const CteOptions options_;
  std::vector<Reader> readers_;
  std::unordered_map<core::PlanNodeId, uint32_t> readerIndices_;
  std::deque<Batch> batches_;
  uint64_t first_{0};
  uint64_t bytes_{0};
  uint64_t peakBytes_{0};
  uint32_t numProducers_{0};
  uint32_t finishedProducers_{0};
  // Monotonic polling state. No payload is accessed through these snapshots.
  std::atomic<uint64_t> published_{0};
  std::atomic<bool> finished_{false};
  std::atomic<bool> allClosed_{false};
  std::atomic<bool> cancelledSnapshot_{false};
};
} // namespace facebook::velox::exec::detail
