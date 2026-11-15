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

#include <optional>
#include "velox/core/PlanNode.h"

namespace facebook::velox::exec {

namespace detail {
struct CteDefinition;
}

struct CteOptions {
  // Optional hard limit per CTE on buffered vector bytes plus estimated
  // per-batch metadata. Fails rather than waiting for readers that may depend
  // on producer completion. Excludes downstream allocations and claimed
  // batches. Unset by default: retained buffers remain charged to their
  // allocating pools and subject to normal query limits and memory arbitration.
  std::optional<uint64_t> maxRetainedBytes;
};

/// Explicit, non-recursive, task-local CTE. The producer is an opaque Velox
/// relational subtree, evaluated once. Each scan() creates a separate reader.
/// All definition and plan node IDs must be unique within the complete query.
class Cte final {
 public:
  Cte(core::PlanNodeId id, core::PlanNodePtr producer);

  core::PlanNodePtr scan(core::PlanNodeId id) const;

 private:
  std::shared_ptr<const detail::CteDefinition> definition_;
};

/// Finalizes a query containing explicit Cte::scan() references. Registers only
/// reachable definitions/readers, including dependencies between CTEs, and adds
/// producer pipelines. Preserves ordinary subplans, keeping a terminal
/// PartitionedOutput above the bindings. Does not infer equivalence or discover
/// sharing from pointer identity. Call once at the complete query root;
/// intermediate branches remain unfinalized.
///
/// No operator whitelist. Subject to normal Velox operator contracts and these
/// execution limits: ungrouped, task-local, no CTE spill/reload.
/// Producers/readers use normal pipeline parallelism; drivers of each reader
/// divide batches, without preserving producer ordering or partitioning.
/// Express required exchanges/sorts using ordinary plan nodes. A tree without
/// CTE references is returned unchanged.
core::PlanNodePtr finalizeCtes(
    const core::PlanNodePtr& query,
    const CteOptions& options = {});

/// Register CTE execution nodes alongside core::PlanNode::registerSerDe().
/// Serialize finalized plans (build()/finalizeCtes()), then deserialize using
/// the normal ISerializable API and its MemoryPool context.
void registerCteSerDe();

} // namespace facebook::velox::exec
