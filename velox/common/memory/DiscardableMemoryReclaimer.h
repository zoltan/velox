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

#include "velox/common/memory/MemoryArbitrator.h"

namespace facebook::velox::memory {

/// Identifies root-pool storage that can be discarded and reconstructed.
/// Reclaimers must preserve pinned and in-flight users and record released
/// storage in Stats::reclaimedBytes. With discardable reclamation enabled,
/// admissions use spare capacity and operator requests prefer these pools
/// before spilling query state.
class DiscardableMemoryReclaimer : public MemoryReclaimer {
 public:
  using MemoryReclaimer::MemoryReclaimer;
};

} // namespace facebook::velox::memory
