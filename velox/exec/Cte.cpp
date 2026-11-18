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
#include "velox/exec/CteInternal.h"

#include <folly/Conv.h>

#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/exec/SharedResultBuffer.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec {
namespace detail {

SharedResultNode::SharedResultNode(
    core::PlanNodeId id,
    core::PlanNodePtr query,
    core::PlanNodePtr producer,
    std::vector<core::PlanNodeId> readers,
    CteOptions options)
    : PlanNode(std::move(id)),
      sources_{std::move(query), std::move(producer)},
      readers_(std::move(readers)),
      options_(options) {}

void SharedResultNode::addDetails(std::stringstream& stream) const {
  stream << readers_.size() << " readers; retention ";
  if (options_.maxRetainedBytes) {
    stream << *options_.maxRetainedBytes << " bytes";
  } else {
    stream << "unlimited";
  }
}

folly::dynamic SharedResultNode::serialize() const {
  auto obj = PlanNode::serialize();
  obj["readers"] = folly::dynamic::array;
  for (const auto& reader : readers_) {
    obj["readers"].push_back(reader);
  }
  // Use a decimal string to preserve the complete uint64_t range in JSON.
  if (options_.maxRetainedBytes) {
    obj["maxRetainedBytes"] = std::to_string(*options_.maxRetainedBytes);
  }
  return obj;
}

core::PlanNodePtr SharedResultNode::create(
    const folly::dynamic& obj,
    void* context) {
  const auto& sources = obj["sources"];
  VELOX_USER_CHECK_EQ(sources.size(), 2, "SharedResult requires two sources");
  std::vector<core::PlanNodeId> readers;
  for (const auto& reader : obj["readers"]) {
    readers.push_back(reader.asString());
  }
  VELOX_USER_CHECK(!readers.empty(), "SharedResult requires readers");
  CteOptions options;
  if (auto* limit = obj.get_ptr("maxRetainedBytes")) {
    options.maxRetainedBytes = folly::to<uint64_t>(limit->asString());
    VELOX_USER_CHECK_GT(
        *options.maxRetainedBytes, 0, "maxRetainedBytes must be positive");
  }
  return std::make_shared<SharedResultNode>(
      obj["id"].asString(),
      ISerializable::deserialize<core::PlanNode>(sources[0], context),
      ISerializable::deserialize<core::PlanNode>(sources[1], context),
      std::move(readers),
      options);
}

SharedScanNode::SharedScanNode(
    core::PlanNodeId id,
    std::shared_ptr<const CteDefinition> definition)
    : PlanNode(std::move(id)),
      sharedId_(definition->id),
      outputType_(definition->producer->outputType()),
      definition_(std::move(definition)) {}

SharedScanNode::SharedScanNode(
    core::PlanNodeId id,
    core::PlanNodeId sharedId,
    RowTypePtr type)
    : PlanNode(std::move(id)),
      sharedId_(std::move(sharedId)),
      outputType_(std::move(type)) {
  VELOX_USER_CHECK_NOT_NULL(outputType_);
}

void SharedScanNode::addDetails(std::stringstream& stream) const {
  stream << sharedId();
}

folly::dynamic SharedScanNode::serialize() const {
  auto obj = PlanNode::serialize();
  obj["sharedId"] = sharedId_;
  obj["outputType"] = outputType_->serialize();
  return obj;
}

core::PlanNodePtr SharedScanNode::create(
    const folly::dynamic& obj,
    void* context) {
  VELOX_USER_CHECK(!obj.count("sources"), "SharedScan cannot have sources");
  return std::make_shared<SharedScanNode>(
      obj["id"].asString(),
      obj["sharedId"].asString(),
      std::dynamic_pointer_cast<const RowType>(
          ISerializable::deserialize<Type>(obj["outputType"])));
}

namespace {

std::shared_ptr<SharedResultBuffer> getBuffer(
    OperatorCtx* ctx,
    const core::PlanNodeId& id) {
  const auto& task = ctx->task();
  auto buffer = std::dynamic_pointer_cast<SharedResultBuffer>(
      task->getCustomJoinBridge(ctx->driverCtx()->splitGroupId, id));
  VELOX_CHECK_NOT_NULL(buffer);
  return buffer;
}

// Sources[0] is the query's ordinary output pipeline. Sources[1] is a separate
// producer pipeline, using the standard custom-join pipeline/bridge hooks.
class SharedResult final : public Operator {
 public:
  SharedResult(int32_t id, DriverCtx* ctx, const SharedResultNode& node)
      : Operator(ctx, node.outputType(), id, node.id(), "SharedResult") {}
  bool needsInput() const override {
    return !noMoreInput_ && !input_;
  }
  void addInput(RowVectorPtr input) override {
    input_ = std::move(input);
  }
  RowVectorPtr getOutput() override {
    return std::exchange(input_, nullptr);
  }
  bool isFinished() override {
    return noMoreInput_ && !input_;
  }
  core::PlanNode::Boundary planNodeBoundary() const override {
    return core::PlanNode::Boundary::kOutput;
  }
  BlockingReason isBlocked(ContinueFuture*) override {
    return BlockingReason::kNotBlocked;
  }
};

class SharedResultSink final : public Operator {
 public:
  SharedResultSink(int32_t id, DriverCtx* ctx, const SharedResultNode& node)
      : Operator(ctx, nullptr, id, node.id(), "SharedResultSink") {}
  void initialize() override {
    Operator::initialize();
    auto buffer = getBuffer(operatorCtx_.get(), planNodeId());
    buffer->setProducerCount(operatorCtx_->numPeers() + 1);
    buffer_ = std::move(buffer);
  }
  bool needsInput() const override {
    return !noMoreInput_;
  }
  void addInput(RowVectorPtr input) override {
    if (!receivedInput_) {
      addRuntimeStat("activeProducerDrivers", RuntimeCounter(1));
      receivedInput_ = true;
    }
    buffer_->add(std::move(input), pool());
  }
  RowVectorPtr getOutput() override {
    return nullptr;
  }
  void noMoreInput() override {
    Operator::noMoreInput();
    finishProducer();
  }
  BlockingReason isBlocked(ContinueFuture*) override {
    return BlockingReason::kNotBlocked;
  }
  bool isFinished() override {
    return noMoreInput_ || buffer_->allReadersClosed();
  }
  core::PlanNode::Boundary planNodeBoundary() const override {
    return core::PlanNode::Boundary::kInput;
  }
  void close() override {
    if (buffer_ && !noMoreInput_ && !buffer_->allReadersClosed()) {
      buffer_->cancel();
    } else {
      finishProducer();
    }
    buffer_.reset();
    Operator::close();
  }

 private:
  void finishProducer() {
    if (buffer_ && !finishedProducer_) {
      const auto peakBytes = buffer_->finish();
      finishedProducer_ = true;
      if (peakBytes > 0) {
        addRuntimeStat(
            "peakRetainedBytes",
            RuntimeCounter(peakBytes, RuntimeCounter::Unit::kBytes));
      }
    }
  }

  bool receivedInput_{false};
  bool finishedProducer_{false};
  std::shared_ptr<SharedResultBuffer> buffer_;
};

class SharedScan final : public SourceOperator {
 public:
  SharedScan(int32_t id, DriverCtx* ctx, const SharedScanNode& node)
      : SourceOperator(ctx, node.outputType(), id, node.id(), "SharedScan"),
        sharedId_(node.sharedId()) {}
  void initialize() override {
    SourceOperator::initialize();
    auto buffer = getBuffer(operatorCtx_.get(), sharedId_);
    reader_ = buffer->readerIndex(planNodeId());
    buffer->setReaderDriverCount(reader_, operatorCtx_->numPeers() + 1);
    buffer_ = std::move(buffer);
  }
  RowVectorPtr getOutput() override {
    auto output = buffer_->next(reader_, pool());
    if (output && !producedOutput_) {
      addRuntimeStat("activeReaderDrivers", RuntimeCounter(1));
      producedOutput_ = true;
    }
    return output;
  }
  BlockingReason isBlocked(ContinueFuture* future) override {
    return buffer_->readerBlocked(reader_, future)
        ? BlockingReason::kWaitForProducer
        : BlockingReason::kNotBlocked;
  }
  bool isFinished() override {
    return buffer_->atEnd(reader_);
  }
  void close() override {
    if (buffer_) {
      buffer_->closeReader(reader_);
      buffer_.reset();
    }
    SourceOperator::close();
  }

 private:
  const core::PlanNodeId sharedId_;
  uint32_t reader_{0};
  bool producedOutput_{false};
  std::shared_ptr<SharedResultBuffer> buffer_;
};

class SharedResultTranslator final : public Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<Operator> toOperator(
      DriverCtx* ctx,
      int32_t id,
      const core::PlanNodePtr& node) override {
    if (auto* result = dynamic_cast<const SharedResultNode*>(node.get())) {
      return std::make_unique<SharedResult>(id, ctx, *result);
    }
    if (auto* scan = dynamic_cast<const SharedScanNode*>(node.get())) {
      return std::make_unique<SharedScan>(id, ctx, *scan);
    }
    return nullptr;
  }
  std::unique_ptr<JoinBridge> toJoinBridge(
      const core::PlanNodePtr& node) override {
    if (auto* result = dynamic_cast<const SharedResultNode*>(node.get())) {
      return std::make_unique<SharedResultBuffer>(
          result->readers(), result->options());
    }
    return nullptr;
  }
  OperatorSupplier toOperatorSupplier(const core::PlanNodePtr& node) override {
    if (auto result = std::dynamic_pointer_cast<const SharedResultNode>(node)) {
      return [result](int32_t id, DriverCtx* ctx) {
        return std::make_unique<SharedResultSink>(id, ctx, *result);
      };
    }
    return nullptr;
  }
};

} // namespace

Operator::PlanNodeTranslator& sharedResultTranslator() {
  static SharedResultTranslator translator;
  return translator;
}
void validateCteExecution(const core::PlanFragment& fragment) {
  const auto& plan = fragment.planNode;
  if (!core::PlanNode::findFirstNode(plan.get(), [](const auto* node) {
        return node->template is<SharedResultNode>() ||
            node->template is<SharedScanNode>();
      })) {
    return;
  }
  // Deserialized scans reference bindings by ID, not by in-process definition
  // pointers. Check those references before creating any pipelines or waiters.
  std::unordered_map<core::PlanNodeId, const SharedResultNode*> bindings;
  std::unordered_set<core::PlanNodeId> ids;
  std::unordered_map<core::PlanNodeId, std::unordered_set<core::PlanNodeId>>
      readers;
  std::unordered_map<core::PlanNodeId, std::vector<core::PlanNodeId>>
      dependencies;
  std::vector<const SharedScanNode*> scans;
  std::function<void(const core::PlanNode*, const core::PlanNodeId&)> visit =
      [&](const auto* node, const auto& producer) {
        VELOX_USER_CHECK(!node->id().empty(), "CTE plan IDs must not be empty");
        VELOX_USER_CHECK(
            ids.insert(node->id()).second,
            "Duplicate plan or CTE ID {}",
            node->id());
        if (auto* binding = dynamic_cast<const SharedResultNode*>(node)) {
          bindings.emplace(binding->id(), binding);
          visit(binding->sources()[0].get(), producer);
          visit(binding->sources()[1].get(), binding->id());
        } else if (auto* scan = dynamic_cast<const SharedScanNode*>(node)) {
          scans.push_back(scan);
          readers[scan->sharedId()].insert(scan->id());
          if (!producer.empty()) {
            dependencies[producer].push_back(scan->sharedId());
          }
        } else {
          for (const auto& source : node->sources()) {
            visit(source.get(), producer);
          }
        }
      };
  visit(plan.get(), "");
  for (const auto* scan : scans) {
    auto it = bindings.find(scan->sharedId());
    VELOX_USER_CHECK(it != bindings.end(), "Unbound CTE {}", scan->sharedId());
    VELOX_USER_CHECK(
        scan->outputType()->equivalent(*it->second->sources()[1]->outputType()),
        "CTE scan {} output type does not match producer",
        scan->id());
  }
  for (const auto& [id, binding] : bindings) {
    const std::unordered_set<core::PlanNodeId> expected(
        binding->readers().begin(), binding->readers().end());
    VELOX_USER_CHECK(
        !expected.empty() && expected.size() == binding->readers().size() &&
            expected == readers[id],
        "CTE {} reader IDs do not match scans",
        id);
  }
  std::unordered_map<core::PlanNodeId, bool> visited;
  std::function<void(const core::PlanNodeId&)> checkCycle =
      [&](const auto& id) {
        auto [it, inserted] = visited.emplace(id, false);
        if (!inserted) {
          VELOX_USER_CHECK(it->second, "Recursive CTE {}", id);
          return;
        }
        // References to unordered_map elements survive rehashing during
        // recursion.
        auto& complete = it->second;
        for (const auto& dependency : dependencies[id]) {
          checkCycle(dependency);
        }
        complete = true;
      };
  for (const auto& [id, binding] : bindings) {
    checkCycle(id);
  }
  VELOX_USER_CHECK(
      !fragment.isGroupedExecution(),
      "Shared results do not support grouped execution");
}
} // namespace detail

Cte::Cte(core::PlanNodeId id, core::PlanNodePtr producer) {
  VELOX_USER_CHECK(!id.empty(), "CTE ID must not be empty");
  VELOX_USER_CHECK_NOT_NULL(producer, "CTE has no producer");
  definition_ = std::make_shared<const detail::CteDefinition>(
      detail::CteDefinition{std::move(id), std::move(producer)});
}

core::PlanNodePtr Cte::scan(core::PlanNodeId id) const {
  return std::make_shared<detail::SharedScanNode>(std::move(id), definition_);
}

core::PlanNodePtr finalizeCtes(
    const core::PlanNodePtr& query,
    const CteOptions& options) {
  VELOX_USER_CHECK_NOT_NULL(query);

  struct Definition {
    std::shared_ptr<const detail::CteDefinition> cte;
    std::vector<core::PlanNodeId> readers;
    bool visiting{true};
  };
  std::unordered_map<core::PlanNodeId, Definition> definitions;
  std::vector<core::PlanNodeId> order;
  std::unordered_set<core::PlanNodeId> ids;
  auto reserveId = [&](const core::PlanNodeId& id) {
    VELOX_USER_CHECK(
        ids.insert(id).second,
        "Duplicate plan or CTE ID {}. Create a separate scan for each CTE use",
        id);
  };
  std::function<void(const core::PlanNodePtr&)> visit = [&](const auto& node) {
    VELOX_USER_CHECK(
        !dynamic_cast<const detail::SharedResultNode*>(node.get()),
        "Finalize CTEs only once, at the complete query root");
    reserveId(node->id());
    if (auto* scan = dynamic_cast<const detail::SharedScanNode*>(node.get())) {
      const auto& cte = scan->definition();
      VELOX_USER_CHECK_NOT_NULL(
          cte, "Deserialized CTE plans are already finalized");
      auto [it, inserted] = definitions.try_emplace(
          cte->id, Definition{.cte = cte, .readers = {}, .visiting = true});
      auto& definition = it->second;
      VELOX_USER_CHECK(
          definition.cte == cte,
          "Distinct CTE definitions have ID {}",
          cte->id);
      if (inserted) {
        reserveId(cte->id);
        visit(cte->producer);
        definition.visiting = false;
        order.push_back(cte->id);
      } else {
        VELOX_USER_CHECK(!definition.visiting, "Recursive CTE {}", cte->id);
      }
      definition.readers.push_back(scan->id());
      return;
    }
    for (const auto& source : node->sources()) {
      visit(source);
    }
  };
  visit(query);
  if (order.empty()) {
    return query;
  }
  if (options.maxRetainedBytes) {
    VELOX_USER_CHECK_GT(
        *options.maxRetainedBytes, 0, "maxRetainedBytes must be positive");
  }
  // Output transport detection requires PartitionedOutput to stay at the root
  // of its pipeline. Bind CTEs to its input, preserving all transport metadata.
  auto output =
      std::dynamic_pointer_cast<const core::PartitionedOutputNode>(query);
  auto result = output ? output->sources()[0] : query;
  for (const auto& id : order) {
    auto& definition = definitions.at(id);
    result = std::make_shared<detail::SharedResultNode>(
        id,
        std::move(result),
        definition.cte->producer,
        std::move(definition.readers),
        options);
  }
  return output
      ? core::PartitionedOutputNode::Builder(*output).source(result).build()
      : result;
}
void registerCteSerDe() {
  auto& registry = DeserializationWithContextRegistryForSharedPtr();
  registry.Register("SharedResultNode", detail::SharedResultNode::create);
  registry.Register("SharedScanNode", detail::SharedScanNode::create);
}
} // namespace facebook::velox::exec
