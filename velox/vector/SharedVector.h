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

#include "velox/vector/ComplexVector.h"

namespace facebook::velox {

/// Shares opaque row payloads without interpreting them as CPU child vectors.
/// Implementations return private metadata over immutable reference-counted
/// storage, preserve readiness and retain storage until asynchronous readers
/// finish. Consumers requiring mutable ownership must materialize a copy.
class SharedVector {
 public:
  virtual ~SharedVector() = default;

  /// Publishes a read-only snapshot. The first call requires exclusive access
  /// to this wrapper; subsequent calls on the published snapshot may run
  /// concurrently and must not mutate it. No payload is copied.
  virtual RowVectorPtr shareForRead(memory::MemoryPool* pool) = 0;
};
} // namespace facebook::velox
