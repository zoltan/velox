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

#include "velox/core/PlanFragment.h"
#include "velox/exec/Cte.h"
#include "velox/exec/Operator.h"

namespace facebook::velox::exec {
namespace detail {

struct CteDefinition {
  const core::PlanNodeId id;
  const core::PlanNodePtr producer;
};

// Internal execution nodes. Applications use Cte and finalizeCtes().
class SharedResultNode final : public core::PlanNode {
 public:
  SharedResultNode(
      core::PlanNodeId id,
      core::PlanNodePtr query,
      core::PlanNodePtr producer,
      std::vector<core::PlanNodeId> readers,
      CteOptions options);

  const RowTypePtr& outputType() const override {
    return sources_[0]->outputType();
  }
  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }
  std::string_view name() const override {
    return "SharedResult";
  }
  folly::dynamic serialize() const override;
  static core::PlanNodePtr create(const folly::dynamic& obj, void* context);
  const std::vector<core::PlanNodeId>& readers() const {
    return readers_;
  }
  const CteOptions& options() const {
    return options_;
  }

 private:
  void addDetails(std::stringstream& stream) const override;
  const std::vector<core::PlanNodePtr> sources_;
  const std::vector<core::PlanNodeId> readers_;
  const CteOptions options_;
};

class SharedScanNode final : public core::PlanNode {
 public:
  SharedScanNode(
      core::PlanNodeId id,
      std::shared_ptr<const CteDefinition> definition);
  SharedScanNode(
      core::PlanNodeId id,
      core::PlanNodeId sharedId,
      RowTypePtr type);
  const RowTypePtr& outputType() const override {
    return outputType_;
  }
  const std::vector<core::PlanNodePtr>& sources() const override {
    return sources_;
  }
  std::string_view name() const override {
    return "SharedScan";
  }
  folly::dynamic serialize() const override;
  static core::PlanNodePtr create(const folly::dynamic& obj, void* context);
  const core::PlanNodeId& sharedId() const {
    return sharedId_;
  }
  const std::shared_ptr<const CteDefinition>& definition() const {
    return definition_;
  }

 private:
  void addDetails(std::stringstream& stream) const override;
  const core::PlanNodeId sharedId_;
  const RowTypePtr outputType_;
  // Present only before serialization; execution uses the ID and output type.
  const std::shared_ptr<const CteDefinition> definition_;
  const std::vector<core::PlanNodePtr> sources_;
};
// Validate unsupported execution features once during local planning.
void validateCteExecution(const core::PlanFragment& fragment);

// A built-in translator: no mutable registration during query planning.
Operator::PlanNodeTranslator& sharedResultTranslator();
} // namespace detail
} // namespace facebook::velox::exec
