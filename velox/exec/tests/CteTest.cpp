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
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <limits>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/common/testutil/TempDirectoryPath.h"
#include "velox/exec/Cte.h"
#include "velox/exec/CteInternal.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/ExchangeSource.h"
#include "velox/exec/LocalPlanner.h"
#include "velox/exec/PartitionFunction.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/SharedResultBuffer.h"
#include "velox/exec/Spiller.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"
#include "velox/vector/FlatMapVector.h"
#include "velox/vector/SequenceVector.h"

namespace facebook::velox::exec::test {
namespace {

class CteTest : public HiveConnectorTestBase {
 protected:
  void SetUp() override {
    HiveConnectorTestBase::SetUp();
    window::prestosql::registerAllWindowFunctions();
    Type::registerSerDe();
    core::ITypedExpr::registerSerDe();
    core::PlanNode::registerSerDe();
    registerPartitionFunctionSerDe();
    registerCteSerDe();
  }

  std::unique_ptr<detail::SharedResultBuffer> makeBuffer(
      uint32_t readerCount,
      CteOptions options,
      uint32_t producers = 1,
      uint32_t readerDrivers = 1) {
    std::vector<core::PlanNodeId> readers;
    for (uint32_t i = 0; i < readerCount; ++i) {
      readers.push_back(std::to_string(i));
    }
    auto buffer =
        std::make_unique<detail::SharedResultBuffer>(readers, options);
    if (producers != 0) {
      buffer->setProducerCount(producers);
    }
    if (readerDrivers != 0) {
      for (uint32_t i = 0; i < readers.size(); ++i) {
        buffer->setReaderDriverCount(i, readerDrivers);
      }
    }
    return buffer;
  }

  RowVectorPtr writable(
      RowVectorPtr row,
      memory::MemoryPool* targetPool = nullptr) {
    VectorPtr result = std::move(row);
    BaseVector::ensureWritable(
        SelectivityVector::empty(),
        result->type(),
        targetPool ? targetPool : pool(),
        result);
    return std::static_pointer_cast<RowVector>(result);
  }

  bool waitUntil(const std::function<bool()>& predicate) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate()) {
      if (std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
  }

  uint64_t positions(const Task& task, const std::string& type, bool input) {
    uint64_t count = 0;
    for (const auto& pipeline : task.taskStats().pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        if (op.operatorType == type) {
          count += input ? op.inputPositions : op.outputPositions;
        }
      }
    }
    return count;
  }
};

TEST_F(CteTest, immediateReadAndCopyOnWrite) {
  auto buffer = makeBuffer(2, {1 << 20});
  auto batch = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  ContinueFuture waiting = ContinueFuture::makeEmpty();
  EXPECT_TRUE(buffer->readerBlocked(0, &waiting));
  buffer->add(batch);
  EXPECT_TRUE(waiting.isReady());
  // Readers can consume output before the producer finishes.
  auto first = buffer->next(0, pool());
  EXPECT_EQ(first->size(), 3);
  EXPECT_EQ(first->childAt(0), batch->childAt(0));
  first = writable(std::move(first));
  first->childAt(0)->asFlatVector<int64_t>()->set(0, 99);
  auto second = buffer->next(1, pool());
  EXPECT_EQ(second->childAt(0)->asFlatVector<int64_t>()->valueAt(0), 1);
  EXPECT_FALSE(buffer->atEnd(0));
  buffer->finish();
  EXPECT_TRUE(buffer->atEnd(0));
  EXPECT_TRUE(buffer->atEnd(1));
}

// Consumers receive private metadata but retain the same nested payloads.

TEST_F(CteTest, nestedSharingAndCopyOnWrite) {
  constexpr int kRows = 32;
  auto makeInput = [&]() {
    auto nested = makeRowVector(
        {makeFlatVector<int64_t>(kRows, [](auto i) { return i; }),
         makeFlatVector<std::string>(kRows, [](auto i) {
           return fmt::format("external string payload for row {}", i);
         })});
    nested->childAt(0)->setNull(3, true);
    auto direct = makeRowVector(
        {makeLazyFlatVector<int64_t>(kRows, [](auto i) { return i * 7; }),
         wrapInLazyDictionary(nested->childAt(1))});
    direct->setNull(3, true);
    std::vector<vector_size_t> offsets(kRows);
    std::iota(offsets.begin(), offsets.end(), 0);
    auto keys = makeRowVector(
        {makeFlatVector<int64_t>(kRows, [](auto i) { return i; })});
    return makeRowVector(
        {makeFlatVector<int64_t>(kRows, [](auto i) { return i; }),
         direct,
         makeArrayVector(offsets, nested),
         makeMapVector(offsets, keys, nested)});
  };
  auto expected = std::static_pointer_cast<RowVector>(
      BaseVector::copy(*makeInput(), pool()));
  for (int run = 0; run < 16; ++run) {
    SCOPED_TRACE(run);
    auto buffer = makeBuffer(2, {1 << 20});
    auto published = makeInput();
    auto lazy = std::static_pointer_cast<LazyVector>(
        published->childAt(1)->as<RowVector>()->childAt(0));
    ASSERT_FALSE(lazy->isLoaded());
    buffer->add(published);
    EXPECT_TRUE(lazy->isLoaded());
    buffer->finish();
    const auto* map = published->childAt(3)->as<MapVector>();
    const std::array<const RowVector*, 5> rows{
        published.get(),
        published->childAt(1)->as<RowVector>(),
        published->childAt(2)->as<ArrayVector>()->elements()->as<RowVector>(),
        map->mapKeys()->as<RowVector>(),
        map->mapValues()->as<RowVector>()};
    std::array<std::array<unsigned char, sizeof(RowVector)>, 5> before;
    for (size_t i = 0; i < rows.size(); ++i) {
      std::memcpy(before[i].data(), rows[i], sizeof(RowVector));
    }
    std::promise<void> start;
    auto ready = start.get_future().share();
    auto first = std::async(std::launch::async, [&]() {
      ready.wait();
      return buffer->next(0, pool());
    });
    auto second = std::async(std::launch::async, [&]() {
      ready.wait();
      return buffer->next(1, pool());
    });
    start.set_value();
    auto left = first.get();
    auto right = second.get();
    for (size_t i = 0; i < rows.size(); ++i) {
      std::array<unsigned char, sizeof(RowVector)> after;
      std::memcpy(after.data(), rows[i], sizeof(RowVector));
      EXPECT_EQ(before[i], after) << "Published vector " << i << " mutated";
    }
    EXPECT_NE(left, right);
    EXPECT_EQ(left->childAt(0), published->childAt(0));
    EXPECT_EQ(right->childAt(0), published->childAt(0));
    EXPECT_EQ(
        left->childAt(2)->as<ArrayVector>()->offsets(),
        published->childAt(2)->as<ArrayVector>()->offsets());
    EXPECT_EQ(
        left->childAt(2)
            ->as<ArrayVector>()
            ->elements()
            ->as<RowVector>()
            ->childAt(0),
        right->childAt(2)
            ->as<ArrayVector>()
            ->elements()
            ->as<RowVector>()
            ->childAt(0));
    assertEqualResults({expected, expected}, {left, right});
    left = writable(std::move(left));
    left->childAt(2)
        ->as<ArrayVector>()
        ->elements()
        ->as<RowVector>()
        ->childAt(0)
        ->asFlatVector<int64_t>()
        ->set(0, -1);
    assertEqualResults({expected}, {right});
  }

  PlanBuilder query;
  query.with("nested", [&](PlanBuilder& cte) {
    std::vector<RowVectorPtr> batches;
    for (int i = 0; i < 64; ++i) {
      batches.push_back(makeInput());
    }
    cte.values(batches);
  });
  auto left = query.newPlan().fromCte("nested").planNode();
  auto right = query.newPlan().fromCte("nested").planNode();
  auto plan = query.localPartition({}, {left, right}).build();
  AssertQueryBuilder(plan).maxDrivers(64).assertResults(
      std::vector<RowVectorPtr>(128, expected));
}

TEST_F(CteTest, selectedLazyRowsArePreserved) {
  auto expected =
      makeRowVector({makeNullableFlatVector<int64_t>({1, 3, 5, std::nullopt})});
  for (bool nestedDictionary : {false, true}) {
    SCOPED_TRACE(nestedDictionary);
    auto makeInput = [&](std::vector<vector_size_t>& loaded) {
      VectorPtr lazy = std::make_shared<LazyVector>(
          pool(),
          BIGINT(),
          8,
          std::make_unique<facebook::velox::test::SimpleVectorLoader>(
              [&](RowSet rows) {
                for (auto row : rows) {
                  VELOX_USER_CHECK(
                      row == 1 || row == 3 || row == 5,
                      "Evaluated an excluded lazy row: {}",
                      row);
                  loaded.push_back(row);
                }
                return makeFlatVector<int64_t>(8, [](auto i) { return i; });
              }));
      if (nestedDictionary) {
        lazy = BaseVector::wrapInDictionary(
            nullptr, makeIndices(8, [](auto i) { return 7 - i; }), 8, lazy);
      }
      auto indices = makeIndices(4, [&](auto i) {
        return i == 3 ? 0 : nestedDictionary ? 6 - i * 2 : i * 2 + 1;
      });
      auto nulls = makeNulls(4, [](auto i) { return i == 3; });
      return makeRowVector(
          {BaseVector::wrapInDictionary(nulls, indices, 4, lazy)});
    };
    std::vector<vector_size_t> loaded;
    auto input = makeInput(loaded);
    ASSERT_TRUE(loaded.empty());
    auto buffer = makeBuffer(2, {1 << 20});
    buffer->add(input);
    EXPECT_EQ(loaded, (std::vector<vector_size_t>{1, 3, 5}));
    assertEqualResults(
        {expected, expected},
        {buffer->next(0, pool()), buffer->next(1, pool())});
    EXPECT_EQ(loaded, (std::vector<vector_size_t>{1, 3, 5}));

    // Each execution gets an unloaded fixture. A plain plan and the CTE plan
    // must both succeed even when evaluation of an excluded row would throw.
    for (bool useCte : {false, true}) {
      loaded.clear();
      PlanBuilder query;
      if (useCte) {
        query.with("selected", [&](PlanBuilder& cte) {
          cte.values({makeInput(loaded)});
        });
        query.fromCte("selected");
      } else {
        query.values({makeInput(loaded)});
      }
      ASSERT_TRUE(loaded.empty());
      AssertQueryBuilder(query.project({"c0 + 0 as c0"}).build())
          .maxDrivers(1)
          .assertResults(expected);
      EXPECT_EQ(loaded, (std::vector<vector_size_t>{1, 3, 5}));
    }
  }
}

TEST_F(CteTest, lazyRowMasksAndConstantsArePreserved) {
  enum class Shape { kNullableRow, kAllNullRow, kConstantRow, kScalarConstant };
  for (auto shape :
       {Shape::kNullableRow,
        Shape::kAllNullRow,
        Shape::kConstantRow,
        Shape::kScalarConstant}) {
    SCOPED_TRACE(static_cast<int>(shape));
    const bool scalar = shape == Shape::kScalarConstant;
    const bool constant = scalar || shape == Shape::kConstantRow;
    const auto selected = constant    ? std::vector<vector_size_t>{3}
        : shape == Shape::kAllNullRow ? std::vector<vector_size_t>{}
                                      : std::vector<vector_size_t>{1, 3, 5, 7};
    auto expected = makeRowVector(
        {constant ? makeNullableFlatVector<int64_t>({3, 3, 3})
             : shape == Shape::kAllNullRow
             ? makeNullableFlatVector<int64_t>(
                   std::vector<std::optional<int64_t>>(8))
             : makeNullableFlatVector<int64_t>(
                   {std::nullopt,
                    1,
                    std::nullopt,
                    3,
                    std::nullopt,
                    5,
                    std::nullopt,
                    7})});
    const std::string expression = scalar ? "c0 + 0 as c0" : "c0.c0 + 0 as c0";
    auto makeInput = [&](std::vector<vector_size_t>& loaded) {
      auto lazy = std::make_shared<LazyVector>(
          pool(),
          BIGINT(),
          8,
          std::make_unique<facebook::velox::test::SimpleVectorLoader>(
              [&](RowSet rows) {
                for (auto row : rows) {
                  VELOX_USER_CHECK(
                      std::find(selected.begin(), selected.end(), row) !=
                          selected.end(),
                      "Evaluated an excluded lazy row: {}",
                      row);
                  loaded.push_back(row);
                }
                return makeFlatVector<int64_t>(8, [](auto i) { return i; });
              }));
      if (scalar) {
        // wrapInConstant eagerly inspects a scalar base. The constructor is
        // the supported way to wrap a scalar lazy vector before loading it.
        return makeRowVector(
            {std::make_shared<ConstantVector<int64_t>>(pool(), 3, 3, lazy)});
      }
      auto row = makeRowVector({lazy});
      if (constant) {
        return makeRowVector({BaseVector::wrapInConstant(3, 3, row)});
      }
      for (int i = 0; i < 8; ++i) {
        row->setNull(i, shape == Shape::kAllNullRow || i % 2 == 0);
      }
      return makeRowVector({row});
    };
    std::vector<vector_size_t> loaded;
    auto buffer = makeBuffer(1, {1 << 20});
    auto input = makeInput(loaded);
    ASSERT_TRUE(loaded.empty());
    buffer->add(input);
    EXPECT_EQ(loaded, selected);
    auto output = buffer->next(0, pool());
    AssertQueryBuilder(
        PlanBuilder().values({output}).project({expression}).planNode())
        .assertResults(expected);
    EXPECT_EQ(loaded, selected);

    for (bool useCte : {false, true}) {
      loaded.clear();
      PlanBuilder query;
      if (useCte) {
        query.with("masked", [&](PlanBuilder& cte) {
          cte.values({makeInput(loaded)});
        });
        query.fromCte("masked");
      } else {
        query.values({makeInput(loaded)});
      }
      ASSERT_TRUE(loaded.empty());
      AssertQueryBuilder assertion(query.project({expression}).build());
      // The baseline scalar expression fast path has a separate unloaded
      // constant rewrapping issue. The ROW cases use default configuration.
      if (scalar) {
        assertion.config(core::QueryConfig::kExprEvalFlatNoNulls, false);
      }
      assertion.maxDrivers(1).assertResults(expected);
      EXPECT_EQ(loaded, selected);
    }
  }
}

TEST_F(CteTest, independentlyLoadedLazyConstant) {
  for (bool complex : {false, true}) {
    SCOPED_TRACE(complex);
    const auto type = complex ? TypePtr(ROW({"v"}, {BIGINT()})) : BIGINT();
    int loads = 0;
    auto lazy = std::make_shared<LazyVector>(
        pool(),
        type,
        8,
        std::make_unique<facebook::velox::test::SimpleVectorLoader>(
            [&](RowSet rows) -> VectorPtr {
              ++loads;
              EXPECT_EQ(rows.size(), 1);
              EXPECT_EQ(rows[0], 3);
              auto values =
                  makeFlatVector<int64_t>(8, [](auto i) { return i; });
              if (complex) {
                auto row = makeRowVector({"v"}, {values});
                row->setNull(3, true);
                return row;
              }
              values->setNull(3, true);
              return values;
            }));
    VectorPtr constant;
    if (complex) {
      constant =
          std::make_shared<ConstantVector<ComplexType>>(pool(), 3, 3, lazy);
    } else {
      constant = std::make_shared<ConstantVector<int64_t>>(pool(), 3, 3, lazy);
    }
    auto input = makeRowVector({constant});
    const vector_size_t selected = 3;
    lazy->load(RowSet(&selected, 1), nullptr);
    // Loading through the base does not initialize its constant wrapper.
    auto buffer = makeBuffer(1, {1 << 20});
    buffer->add(input);
    auto output = buffer->next(0, pool());
    auto expected =
        makeRowVector({BaseVector::createNullConstant(type, 3, pool())});
    assertEqualResults({expected}, {output});
    EXPECT_EQ(output->childAt(0), constant); // Null constants stay shared.
    EXPECT_EQ(loads, 1);
  }
}

TEST_F(CteTest, flatMapWithIndependentlyLoadedConstant) {
  auto lazy = makeLazyFlatVector<int64_t>(
      8, [](auto i) { return i; }, [](auto i) { return i == 3; });
  auto constant = std::make_shared<ConstantVector<int64_t>>(pool(), 3, 3, lazy);
  auto map = std::make_shared<FlatMapVector>(
      pool(),
      MAP(BIGINT(), BIGINT()),
      nullptr,
      3,
      makeFlatVector<int64_t>({1}),
      std::vector<VectorPtr>{constant},
      std::vector<BufferPtr>{});
  const vector_size_t selected = 3;
  lazy->load(RowSet(&selected, 1), nullptr);
  auto buffer = makeBuffer(1, {1 << 20});
  buffer->add(makeRowVector({map}));
  auto output = buffer->next(0, pool());
  EXPECT_EQ(output->childAt(0)->as<FlatMapVector>()->mapValuesAt(0), constant);
  auto plan = PlanBuilder()
                  .values({output})
                  .project({"element_at(c0, cast(1 as bigint)) + 0 as c0"})
                  .planNode();
  AssertQueryBuilder(plan).assertResults(
      makeRowVector({makeNullableFlatVector<int64_t>(
          {std::nullopt, std::nullopt, std::nullopt})}));
}

TEST_F(CteTest, partiallyProjectedLazyRows) {
  for (bool nested : {false, true}) {
    for (bool useCte : {false, true}) {
      SCOPED_TRACE(fmt::format("nested={}, cte={}", nested, useCte));
      std::vector<vector_size_t> loaded;
      auto lazy = std::make_shared<LazyVector>(
          pool(),
          BIGINT(),
          8,
          std::make_unique<facebook::velox::test::SimpleVectorLoader>(
              [&](RowSet rows) {
                loaded.assign(rows.begin(), rows.end());
                return makeFlatVector<int64_t>(8, [](auto i) { return i; });
              }));
      auto partial = std::make_shared<RowVector>(
          pool(),
          ROW({"unused0", "v", "unused1"}, {BIGINT(), BIGINT(), BIGINT()}),
          nullptr,
          8,
          std::vector<VectorPtr>{nullptr, lazy, nullptr});
      if (nested) {
        partial->setNull(3, true);
      }
      auto input = nested ? makeRowVector({partial}) : partial;
      auto expected = makeRowVector({makeFlatVector<int64_t>(
          8,
          [](auto i) { return i; },
          [nested](auto i) { return nested && i == 3; })});
      PlanBuilder query;
      if (useCte) {
        query.with("partial", [&](PlanBuilder& cte) { cte.values({input}); });
        query.fromCte("partial");
      } else {
        query.values({input});
      }
      auto plan =
          query.project({nested ? "c0.v + 0 as c0" : "v + 0 as c0"}).build();
      AssertQueryBuilder(plan).maxDrivers(64).assertResults(expected);
      EXPECT_EQ(
          loaded,
          nested ? (std::vector<vector_size_t>{0, 1, 2, 4, 5, 6, 7})
                 : (std::vector<vector_size_t>{0, 1, 2, 3, 4, 5, 6, 7}));
    }
  }
}

TEST_F(CteTest, normalFlatMapProjectionKeepsOtherKeysLazy) {
  auto unused = std::make_shared<LazyVector>(
      pool(),
      BIGINT(),
      8,
      std::make_unique<facebook::velox::test::SimpleVectorLoader>(
          [&](RowSet) -> VectorPtr {
            ADD_FAILURE() << "A normal key projection loaded an unused key";
            return makeFlatVector<int64_t>(8, [](auto i) { return i; });
          }));
  auto values = makeLazyFlatVector<int64_t>(8, [](auto i) { return i; });
  auto map = std::make_shared<FlatMapVector>(
      pool(),
      MAP(BIGINT(), BIGINT()),
      nullptr,
      8,
      makeFlatVector<int64_t>({1, 2}),
      std::vector<VectorPtr>{values, unused},
      std::vector<BufferPtr>{});
  auto plan = PlanBuilder()
                  .values({makeRowVector({map})})
                  .project({"element_at(c0, cast(1 as bigint)) + 0 as c0"})
                  .planNode();
  AssertQueryBuilder(plan).assertResults(
      makeRowVector({makeFlatVector<int64_t>(8, [](auto i) { return i; })}));
  EXPECT_FALSE(unused->isLoaded());
}

TEST_F(CteTest, lazyFlatMapSelections) {
  for (int wrapper = 0; wrapper < 4; ++wrapper) {
    for (bool useCte : {false, true}) {
      SCOPED_TRACE(fmt::format("wrapper={}, cte={}", wrapper, useCte));
      const auto selected = wrapper == 0 ? std::vector<vector_size_t>{1, 5, 7}
          : wrapper == 2                 ? std::vector<vector_size_t>{5}
                                         : std::vector<vector_size_t>{1, 5};
      std::vector<vector_size_t> loaded;
      auto lazy = std::make_shared<LazyVector>(
          pool(),
          BIGINT(),
          8,
          std::make_unique<facebook::velox::test::SimpleVectorLoader>(
              [&](RowSet rows) {
                for (auto row : rows) {
                  if (useCte) {
                    VELOX_USER_CHECK(
                        std::find(selected.begin(), selected.end(), row) !=
                            selected.end(),
                        "Evaluated excluded FlatMap value {}",
                        row);
                  }
                  loaded.push_back(row);
                }
                return makeFlatVector<int64_t>(8, [](auto i) { return i; });
              }));
      VectorPtr map = std::make_shared<FlatMapVector>(
          pool(),
          MAP(BIGINT(), BIGINT()),
          makeNulls(8, [](auto i) { return i == 3; }),
          8,
          makeFlatVector<int64_t>({1}),
          std::vector<VectorPtr>{lazy},
          std::vector<BufferPtr>{
              makeNulls(8, [](auto i) { return i % 2 == 0; })});
      std::vector<std::optional<int64_t>> result{
          std::nullopt,
          1,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          5,
          std::nullopt,
          7};
      if (wrapper == 1) {
        map = BaseVector::wrapInDictionary(
            makeNulls(5, [](auto i) { return i == 0; }),
            makeIndices({7, 3, 5, 1, 0}),
            5,
            map);
        result = {std::nullopt, std::nullopt, 5, 1, std::nullopt};
      } else if (wrapper == 2) {
        map = BaseVector::wrapInConstant(3, 5, map);
        result = {5, 5, 5};
      } else if (wrapper == 3) {
        auto row = makeRowVector({map});
        row->setNull(7, true);
        map = row;
        result[7] = std::nullopt;
      }
      auto input = makeRowVector({map});
      auto expected = makeRowVector({makeNullableFlatVector<int64_t>(result)});
      ASSERT_TRUE(loaded.empty());
      PlanBuilder query;
      if (useCte) {
        query.with("flat_map", [&](PlanBuilder& cte) { cte.values({input}); });
        query.fromCte("flat_map");
      } else {
        query.values({input});
      }
      const auto field = wrapper == 3 ? "c0.c0" : "c0";
      auto plan =
          query
              .project({fmt::format(
                  "element_at({}, cast(1 as bigint)) + 0 as c0", field)})
              .build();
      AssertQueryBuilder(plan).maxDrivers(1).assertResults(expected);
      if (useCte) {
        EXPECT_EQ(loaded, selected);
      }
    }
  }
}

TEST_F(CteTest, flatMapRetainedByteLimit) {
  constexpr int kRows = 1024;
  auto keys = makeFlatVector<int64_t>({1});
  auto values = makeFlatVector<int64_t>(kRows, [](auto i) { return i; });
  auto inMap = makeNulls(kRows, [](auto i) { return i % 2 == 0; });
  auto inner = std::make_shared<FlatMapVector>(
      pool(),
      MAP(BIGINT(), BIGINT()),
      nullptr,
      kRows,
      keys,
      std::vector<VectorPtr>{values},
      std::vector<BufferPtr>{inMap});
  auto outer = std::make_shared<FlatMapVector>(
      pool(),
      MAP(BIGINT(), inner->type()),
      nullptr,
      kRows,
      keys,
      std::vector<VectorPtr>{inner},
      std::vector<BufferPtr>{});
  for (const auto& map : {inner, outer}) {
    SCOPED_TRACE(map->type()->toString());
    auto batch = makeRowVector({map});
    const auto bytes = detail::SharedResultBuffer::batchBytes(*batch);
    ASSERT_GT(bytes, 4096);
    auto tiny = makeBuffer(1, {4096});
    VELOX_ASSERT_THROW(tiny->add(batch), "retained-byte limit exceeded");
    auto buffer = makeBuffer(2, {bytes});
    buffer->add(batch);
    VELOX_ASSERT_THROW(buffer->add(batch), "retained-byte limit exceeded");
    auto first = buffer->next(0, pool());
    EXPECT_EQ(first->childAt(0)->as<FlatMapVector>()->distinctKeys(), keys);
    facebook::velox::test::assertEqualVectors(batch, first);
    facebook::velox::test::assertEqualVectors(batch, buffer->next(1, pool()));
    buffer->add(batch); // Releasing the queue entry restores its budget.
  }
}

TEST_F(CteTest, producerReuseDetachesPublishedPayload) {
  auto input = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  auto expected = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  auto buffer = makeBuffer(2, {1 << 20});
  buffer->add(input);
  // No test-owned aliases to the producer row or its children can force COW.
  ASSERT_EQ(input.use_count(), 1);
  ASSERT_EQ(input->childAt(0).use_count(), 2);
  input = writable(std::move(input));
  input->childAt(0)->asFlatVector<int64_t>()->set(0, 99);
  buffer->finish();
  assertEqualResults(
      {expected, expected}, {buffer->next(0, pool()), buffer->next(1, pool())});
}

TEST_F(CteTest, sharedPayloadLifetime) {
  auto input = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  std::weak_ptr<BaseVector> leaf = input->childAt(0);
  const auto* data = input->childAt(0)->asFlatVector<int64_t>()->rawValues();
  auto buffer = makeBuffer(2, {1 << 20});
  buffer->add(input);
  input.reset();
  auto left = buffer->next(0, pool());
  auto right = buffer->next(1, pool());
  EXPECT_EQ(left->childAt(0)->asFlatVector<int64_t>()->rawValues(), data);
  EXPECT_EQ(right->childAt(0)->asFlatVector<int64_t>()->rawValues(), data);
  buffer->finish();
  buffer.reset(); // The queue is no longer an owner; downstream still is.
  EXPECT_FALSE(leaf.expired());
  left.reset();
  EXPECT_FALSE(leaf.expired());
  right = writable(std::move(right)); // Last owner can write without copying.
  EXPECT_EQ(right->childAt(0)->asFlatVector<int64_t>()->rawValues(), data);
  right.reset();
  EXPECT_TRUE(leaf.expired());
}

TEST_F(CteTest, encodedPayloadsStayShared) {
  auto numbers = makeFlatVector<int64_t>({10, 20, 30});
  auto nested = makeRowVector({numbers});
  auto indices = makeIndices(6, [](auto i) { return i % 3; });
  auto dictionary = BaseVector::wrapInDictionary(nullptr, indices, 6, nested);
  auto strings = makeFlatVector<std::string>(
      {"constant string longer than inline storage"});
  auto constant = BaseVector::wrapInConstant(6, 0, strings);
  auto booleans = makeFlatVector<bool>(
      6, [](auto i) { return i % 2 == 0; }, [](auto i) { return i == 3; });
  auto nullRow =
      BaseVector::createNullConstant(ROW({"v"}, {BIGINT()}), 6, pool());
  auto input = makeRowVector({dictionary, constant, booleans, nullRow});
  auto buffer = makeBuffer(2, {1 << 20});
  buffer->add(input);
  buffer->finish();
  auto left = buffer->next(0, pool());
  auto right = buffer->next(1, pool());
  for (const auto& rows : {left, right}) {
    EXPECT_EQ(rows->childAt(0)->encoding(), VectorEncoding::Simple::DICTIONARY);
    EXPECT_EQ(rows->childAt(0)->wrapInfo(), indices);
    EXPECT_EQ(
        rows->childAt(0)->valueVector()->as<RowVector>()->childAt(0), numbers);
    EXPECT_EQ(rows->childAt(1), constant);
    EXPECT_EQ(rows->childAt(2), booleans);
    EXPECT_EQ(rows->childAt(3), nullRow);
  }
  EXPECT_NE(left->childAt(0), right->childAt(0));
  assertEqualResults({input, input}, {left, right});
  auto expected =
      std::static_pointer_cast<RowVector>(BaseVector::copy(*input, pool()));
  booleans.reset(); // Only the producer and shared result own this payload.
  input = writable(std::move(input)); // Producer reuse must also detach.
  input->childAt(2)->asFlatVector<bool>()->set(0, false);
  assertEqualResults({expected, expected}, {left, right});
}

TEST_F(CteTest, sequenceReadersHaveIndependentCursors) {
  auto values = makeFlatVector<int64_t>({10, 20, 30});
  auto lengths = AlignedBuffer::allocate<vector_size_t>(3, pool());
  auto* raw = lengths->asMutable<vector_size_t>();
  raw[0] = raw[1] = raw[2] = 100;
  auto sequence =
      std::make_shared<SequenceVector<int64_t>>(pool(), 300, values, lengths);
  auto input = makeRowVector({sequence});
  auto buffer = makeBuffer(2, {1 << 20});
  buffer->add(input);
  buffer->finish();
  auto left = buffer->next(0, pool());
  auto right = buffer->next(1, pool());
  EXPECT_NE(left->childAt(0), right->childAt(0));
  EXPECT_EQ(left->childAt(0)->wrapInfo(), lengths);
  EXPECT_EQ(right->childAt(0)->valueVector(), values);
  auto read = [](const RowVectorPtr& rows, bool reverse) {
    const auto* seq = rows->childAt(0)->as<SequenceVector<int64_t>>();
    for (int run = 0; run < 100; ++run) {
      for (int i = 0; i < 300; ++i) {
        const auto index = reverse ? 299 - i : i;
        EXPECT_EQ(seq->valueAt(index), (index / 100 + 1) * 10);
      }
    }
  };
  auto first = std::async(std::launch::async, [&] { read(left, false); });
  auto second = std::async(std::launch::async, [&] { read(right, true); });
  first.get();
  second.get();
}

TEST_F(CteTest, sequenceResultsFromLazyLoaders) {
  for (bool complex : {false, true}) {
    SCOPED_TRACE(complex);
    auto payload = makeFlatVector<int64_t>({10, 20, 30});
    auto lengths = makeIndices({2, 2, 2});
    VectorPtr sequence;
    if (complex) {
      sequence = std::make_shared<SequenceVector<ComplexType>>(
          pool(), 6, makeRowVector({payload}), lengths);
    } else {
      sequence = std::make_shared<SequenceVector<int64_t>>(
          pool(), 6, payload, lengths);
    }
    for (bool lazy : {false, true}) {
      VectorPtr input = sequence;
      if (lazy) {
        input = std::make_shared<LazyVector>(
            pool(),
            sequence->type(),
            6,
            std::make_unique<facebook::velox::test::SimpleVectorLoader>(
                [sequence](RowSet) { return sequence; }));
      }
      auto buffer = makeBuffer(2, {1 << 20});
      buffer->add(makeRowVector({input}));
      buffer->finish();
      for (int reader = 0; reader < 2; ++reader) {
        auto output = buffer->next(reader, pool())->childAt(0);
        EXPECT_EQ(output->encoding(), VectorEncoding::Simple::SEQUENCE);
        EXPECT_EQ(output->wrapInfo(), lengths);
        EXPECT_EQ(output->wrappedIndex(5), 2);
        const auto& values = output->valueVector();
        EXPECT_EQ(
            complex ? values->as<RowVector>()->childAt(0) : values, payload);
      }
    }
  }
}

TEST_F(CteTest, retentionPeakReportedByFinalProducer) {
  auto batch = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  const auto bytes = detail::SharedResultBuffer::batchBytes(*batch);
  auto buffer = makeBuffer(2, {4 * bytes}, 2);
  buffer->add(batch);
  buffer->add(batch);
  buffer->closeReader(0);
  buffer->closeReader(1);
  EXPECT_EQ(buffer->finish(), 0);
  EXPECT_EQ(buffer->finish(), 2 * bytes);
}

TEST_F(CteTest, retainedByteLimitAndRelease) {
  auto batch = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  const auto bytes = detail::SharedResultBuffer::batchBytes(*batch);
  auto buffer = makeBuffer(2, {bytes});
  buffer->add(batch);
  EXPECT_NE(buffer->next(0, pool()), nullptr);
  // A delayed reader pins the batch. Fail at the cap instead of blocking a
  // producer on a reader that may depend on its completion.
  VELOX_ASSERT_THROW(
      buffer->add(batch), "Shared result retained-byte limit exceeded");
  EXPECT_NE(buffer->next(1, pool()), nullptr);
  buffer->add(batch); // Every reader claimed the first batch: budget is freed.
  buffer->finish();
  EXPECT_NE(buffer->next(1, pool()), nullptr);
  EXPECT_TRUE(buffer->atEnd(1));
}

TEST_F(CteTest, earlyCloseAndCancellation) {
  auto batch = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  auto buffer = makeBuffer(2, {detail::SharedResultBuffer::batchBytes(*batch)});
  buffer->add(batch);
  buffer->next(0, pool());
  buffer->closeReader(1);
  buffer->add(batch); // Closing the delayed reader releases the old batch.
  buffer->closeReader(0);
  EXPECT_TRUE(buffer->allReadersClosed());

  auto cancelled = makeBuffer(2, {1 << 20});
  cancelled->add(batch);
  cancelled->next(0, pool());
  ContinueFuture reader = ContinueFuture::makeEmpty();
  EXPECT_TRUE(cancelled->readerBlocked(0, &reader));
  // Cancellation must dispatch through the same base pointer Task holds.
  JoinBridge* bridge = cancelled.get();
  bridge->cancel();
  EXPECT_TRUE(reader.isReady());
  VELOX_ASSERT_THROW(cancelled->next(0, pool()), "Shared result cancelled");

  auto empty = makeBuffer(2, {1 << 20});
  EXPECT_TRUE(empty->readerBlocked(0, &reader));
  empty->finish();
  EXPECT_TRUE(reader.isReady());
  EXPECT_TRUE(empty->atEnd(0));
  EXPECT_TRUE(empty->atEnd(1));
}

TEST_F(CteTest, parallelCompletionAndLateInitialization) {
  auto buffer = makeBuffer(1, {1 << 20}, 0, 0);
  buffer->setProducerCount(64);
  buffer->setReaderDriverCount(0, 64);
  ContinueFuture waiting = ContinueFuture::makeEmpty();
  EXPECT_TRUE(buffer->readerBlocked(0, &waiting));
  for (int i = 0; i < 63; ++i) {
    buffer->finish();
  }
  EXPECT_FALSE(buffer->atEnd(0));
  EXPECT_FALSE(waiting.isReady());
  // A driver first initialized after its peers finish must not reset counts.
  buffer->setProducerCount(64);
  auto batch = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  buffer->add(batch);
  EXPECT_TRUE(waiting.isReady());
  for (int i = 0; i < 63; ++i) {
    buffer->closeReader(0);
  }
  EXPECT_FALSE(buffer->allReadersClosed());
  buffer->setReaderDriverCount(0, 64);
  EXPECT_EQ(buffer->next(0, pool())->size(), 3);
  EXPECT_FALSE(buffer->atEnd(0));
  buffer->finish();
  EXPECT_TRUE(buffer->atEnd(0));
  buffer->closeReader(0);
  EXPECT_TRUE(buffer->allReadersClosed());
}

TEST_F(CteTest, concurrentProducersAndReaders) {
  constexpr int kProducers = 4;
  constexpr int kDriversPerReader = 8;
  constexpr int kBatches = 256;
  constexpr int kRows = 128;
  std::vector<RowVectorPtr> batches;
  for (int batch = 0; batch < kBatches; ++batch) {
    batches.push_back(makeRowVector({makeFlatVector<int64_t>(
        kRows, [batch](auto i) { return batch * kRows + i; })}));
  }
  // Each logical reader distributes work across eight concurrent drivers.
  auto buffer = makeBuffer(2, {8 << 20}, 0, 0);
  buffer->setProducerCount(kProducers);
  std::vector<std::future<std::vector<int64_t>>> readers[2];
  for (int reader = 0; reader < 2; ++reader) {
    buffer->setReaderDriverCount(reader, kDriversPerReader);
    for (int driver = 0; driver < kDriversPerReader; ++driver) {
      readers[reader].push_back(std::async(std::launch::async, [&, reader]() {
        std::vector<int64_t> rows;
        try {
          while (!buffer->atEnd(reader)) {
            ContinueFuture future = ContinueFuture::makeEmpty();
            if (buffer->readerBlocked(reader, &future)) {
              std::move(future).wait();
              continue;
            }
            if (auto batch = buffer->next(reader, pool())) {
              batch = writable(std::move(batch));
              auto* values = batch->childAt(0)->asFlatVector<int64_t>();
              for (int i = 0; i < batch->size(); ++i) {
                rows.push_back(values->valueAt(i));
                values->set(i, -1); // Must not mutate another reader's input.
              }
            }
          }
          buffer->closeReader(reader);
        } catch (...) {
          buffer->cancel();
          throw;
        }
        return rows;
      }));
    }
  }
  std::vector<std::future<void>> producers;
  for (int producer = 0; producer < kProducers; ++producer) {
    producers.push_back(std::async(std::launch::async, [&, producer]() {
      try {
        for (int batch = producer; batch < kBatches; batch += kProducers) {
          buffer->add(batches[batch]);
        }
        buffer->finish();
      } catch (...) {
        buffer->cancel();
        throw;
      }
    }));
  }
  for (auto& producer : producers) {
    producer.get();
  }
  for (auto& reader : readers) {
    std::vector<int64_t> rows;
    for (auto& driver : reader) {
      auto result = driver.get();
      rows.insert(rows.end(), result.begin(), result.end());
    }
    std::sort(rows.begin(), rows.end());
    ASSERT_EQ(rows.size(), kBatches * kRows);
    for (int i = 0; i < rows.size(); ++i) {
      ASSERT_EQ(rows[i], i);
    }
  }
  EXPECT_TRUE(buffer->allReadersClosed());
}

TEST_F(CteTest, publicationOverlapsCancellationAndReaderClose) {
  for (bool cancel : {false, true}) {
    SCOPED_TRACE(cancel);
    auto buffer = makeBuffer(2, {1 << 20}, 2);
    std::promise<void> loading;
    auto loadingStarted = loading.get_future();
    std::promise<void> resume;
    auto resumeLoading = resume.get_future().share();
    auto lazy = std::make_shared<LazyVector>(
        pool(),
        BIGINT(),
        3,
        std::make_unique<facebook::velox::test::SimpleVectorLoader>(
            [&](RowSet) {
              loading.set_value();
              resumeLoading.wait();
              return makeFlatVector<int64_t>({4, 5, 6});
            }));
    std::future<void> publishing;
    std::future<void> consuming;
    bool resumed = false;
    SCOPE_EXIT {
      if (!resumed) {
        resume.set_value();
      }
    };
    publishing = std::async(
        std::launch::async, [&] { buffer->add(makeRowVector({lazy})); });
    ASSERT_EQ(
        loadingStarted.wait_for(std::chrono::seconds(5)),
        std::future_status::ready);

    // While the first producer is loading, another publishes and two readers
    // claim its batch. Loading cannot hold the queue mutex.
    auto batch = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
    consuming = std::async(std::launch::async, [&, batch] {
      buffer->add(batch);
      assertEqualResults(
          {batch, batch}, {buffer->next(0, pool()), buffer->next(1, pool())});
    });
    ASSERT_EQ(
        consuming.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    consuming.get();
    ContinueFuture first = ContinueFuture::makeEmpty();
    ContinueFuture second = ContinueFuture::makeEmpty();
    ASSERT_TRUE(buffer->readerBlocked(0, &first));
    ASSERT_TRUE(buffer->readerBlocked(1, &second));
    if (cancel) {
      buffer->cancel();
    } else {
      buffer->closeReader(0);
      buffer->closeReader(1);
    }
    EXPECT_TRUE(first.isReady());
    EXPECT_TRUE(second.isReady());
    resume.set_value();
    resumed = true;
    ASSERT_EQ(
        publishing.wait_for(std::chrono::seconds(5)),
        std::future_status::ready);
    if (cancel) {
      VELOX_ASSERT_THROW(publishing.get(), "Shared result cancelled");
      VELOX_ASSERT_THROW(buffer->next(0, pool()), "Shared result cancelled");
      buffer->closeReader(0);
      buffer->closeReader(1);
    } else {
      EXPECT_NO_THROW(publishing.get());
      EXPECT_EQ(buffer->next(0, pool()), nullptr);
    }
    buffer->finish();
    buffer->finish();
    EXPECT_TRUE(buffer->allReadersClosed());
  }
}

TEST_F(CteTest, releasedPayloadCallbacksRunOutsideLock) {
  struct Releaser {
    BufferPtr storage;
    std::function<void()> callback;
    void addRef() const {}
    void release() const {
      callback();
    }
  };
  auto buffer = makeBuffer(2, {1 << 20});
  std::vector<std::future<bool>> callbacks;
  for (int i = 0; i < 3; ++i) {
    auto storage = AlignedBuffer::allocate<int64_t>(1, pool(), i);
    auto view = BufferView<Releaser>::create(
        storage,
        Releaser{storage, [&] {
                   // Keep the future alive until closeReader returns. If
                   // destruction ever moves back under the lock, report a
                   // bounded failure instead of hanging in the future's
                   // destructor while the callback still owns the lock.
                   callbacks.push_back(std::async(std::launch::async, [&] {
                     return buffer->allReadersClosed();
                   }));
                   EXPECT_EQ(
                       callbacks.back().wait_for(std::chrono::seconds(5)),
                       std::future_status::ready);
                 }});
    auto values = std::make_shared<FlatVector<int64_t>>(
        pool(),
        BIGINT(),
        nullptr,
        1,
        std::move(view),
        std::vector<BufferPtr>{});
    buffer->add(makeRowVector({std::move(values)}));
  }
  buffer->closeReader(0);
  EXPECT_TRUE(callbacks.empty()); // The other reader still pins all batches.
  buffer->closeReader(1);
  ASSERT_EQ(callbacks.size(), 3);
  for (auto& callback : callbacks) {
    EXPECT_TRUE(callback.get());
  }
}

TEST_F(CteTest, zeroColumnBatchesRespectBudgets) {
  auto batch = std::make_shared<RowVector>(
      pool(), ROW({}, {}), nullptr, 1024, std::vector<VectorPtr>{});
  ASSERT_EQ(batch->retainedSize(), 0);
  const auto bytes = detail::SharedResultBuffer::batchBytes(*batch);
  ASSERT_GT(bytes, 0);
  auto retained = makeBuffer(1, {bytes});
  retained->add(batch);
  VELOX_ASSERT_THROW(retained->add(batch), "retained-byte limit exceeded");
  auto tiny = makeBuffer(1, {1});
  VELOX_ASSERT_THROW(tiny->add(batch), "retained-byte limit exceeded");

  PlanBuilder query;
  query.with("empty_columns", [&](PlanBuilder& cte) { cte.values({batch}); });
  query.fromCte("empty_columns")
      .localPartition({})
      .singleAggregation({}, {"count(*)"});
  AssertQueryBuilder(query.build())
      .maxDrivers(64)
      .assertResults(makeRowVector({makeFlatVector<int64_t>({1024})}));
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(query.build({1})).copyResults(pool()),
      "retained-byte limit exceeded");
}

TEST_F(CteTest, notificationsMatchChangedConditions) {
  auto batch = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  auto buffer = makeBuffer(2, {1 << 20}, 1, 2);
  ContinueFuture a = ContinueFuture::makeEmpty();
  ContinueFuture b = ContinueFuture::makeEmpty();
  ContinueFuture other = ContinueFuture::makeEmpty();
  ASSERT_TRUE(buffer->readerBlocked(0, &a));
  ASSERT_TRUE(buffer->readerBlocked(0, &b));
  ASSERT_TRUE(buffer->readerBlocked(1, &other));
  buffer->add(batch);
  EXPECT_TRUE(a.isReady());
  EXPECT_FALSE(b.isReady()); // Only one new batch for this group.
  EXPECT_TRUE(other.isReady());
  buffer->next(0, pool());
  ContinueFuture noData = ContinueFuture::makeEmpty();
  ASSERT_TRUE(buffer->readerBlocked(0, &noData));
  buffer->next(1, pool());
  buffer->closeReader(0);
  EXPECT_FALSE(b.isReady());
  EXPECT_FALSE(
      noData.isReady()); // Claims and partial close don't publish data.
  buffer->finish();
  EXPECT_TRUE(b.isReady());
  EXPECT_TRUE(noData.isReady());
}

TEST_F(CteTest, partialCloseHandsUnreadBatchToPeer) {
  auto buffer = makeBuffer(1, {1 << 20}, 1, 2);
  ContinueFuture a = ContinueFuture::makeEmpty();
  ContinueFuture b = ContinueFuture::makeEmpty();
  ASSERT_TRUE(buffer->readerBlocked(0, &a));
  ASSERT_TRUE(buffer->readerBlocked(0, &b));
  auto batch = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  buffer->add(batch);
  ASSERT_TRUE(a.isReady());
  ASSERT_FALSE(b.isReady());
  buffer->closeReader(0); // The notified driver exits without claiming data.
  EXPECT_TRUE(b.isReady());
  EXPECT_FALSE(buffer->allReadersClosed());
  assertEqualResults({batch}, {buffer->next(0, pool())});
  buffer->finish();
  EXPECT_TRUE(buffer->atEnd(0));
  buffer->closeReader(0);
  EXPECT_TRUE(buffer->allReadersClosed());
}

TEST_F(CteTest, partitionedOutputRemainsTerminal) {
  auto factories = ExchangeSource::factories();
  SCOPE_EXIT {
    ExchangeSource::factories() = std::move(factories);
  };
  ExchangeSource::registerFactory(createLocalExchangeSource);
  auto input = makeRowVector(
      {makeFlatVector<int64_t>(256, [](auto i) { return i; }),
       makeFlatVector<int64_t>(256, [](auto i) { return i * 3; })});
  auto expected = makeRowVector({input->childAt(1), input->childAt(0)});
  for (bool broadcast : {false, true}) {
    for (int drivers : {1, 64}) {
      PlanBuilder query;
      query.with("base", [&](PlanBuilder& cte) { cte.values({input}); });
      query.with("dependent", [](PlanBuilder& cte) { cte.fromCte("base"); });
      query.fromCte("dependent");
      if (broadcast) {
        query.partitionedOutputBroadcast({"c1", "c0"});
      } else {
        query.partitionedOutput(
            {"c0"}, 2, std::vector<std::string>{"c1", "c0"});
      }
      auto plan = ISerializable::deserialize<core::PlanNode>(
          folly::parseJson(folly::toJson(query.build()->serialize())), pool());
      auto output =
          std::dynamic_pointer_cast<const core::PartitionedOutputNode>(plan);
      ASSERT_NE(output, nullptr);
      EXPECT_EQ(plan->id(), query.planNode()->id());
      EXPECT_EQ(
          plan->outputType()->names(), query.planNode()->outputType()->names());
      const auto id =
          fmt::format("local://cte-output-{}-{}", broadcast, drivers);
      auto ctx =
          core::QueryCtx::Builder().executor(driverExecutor_.get()).build();
      auto task = Task::create(
          id, core::PlanFragment{plan}, 0, ctx, Task::ExecutionMode::kParallel);
      SCOPE_EXIT {
        if (task->isRunning()) {
          task->requestCancel().wait();
        }
      };
      task->start(drivers);
      if (broadcast) {
        task->updateOutputBuffers(2, true);
      }
      std::vector<RowVectorPtr> results;
      for (int destination = 0; destination < 2; ++destination) {
        auto consumer = PlanBuilder()
                            .exchange(plan->outputType(), output->serdeKind())
                            .planNode();
        results.push_back(AssertQueryBuilder(consumer)
                              .destination(destination)
                              .split(std::make_shared<RemoteConnectorSplit>(id))
                              .copyResults(pool()));
      }
      assertEqualResults(
          std::vector<RowVectorPtr>(broadcast ? 2 : 1, expected), results);
      EXPECT_TRUE(waitForTaskCompletion(task.get()));
      EXPECT_EQ(positions(*task, "SharedResultSink", true), 512);
    }
  }
}

TEST_F(CteTest, groupedExecutionRejectedDuringPlanning) {
  PlanBuilder query;
  core::PlanNodeId scanId;
  query.with("grouped", [&](PlanBuilder& cte) {
    cte.tableScan(ROW({"c0"}, {BIGINT()})).capturePlanNodeId(scanId);
  });
  core::PlanFragment fragment{query.fromCte("grouped").build()};
  fragment.executionStrategy = core::ExecutionStrategy::kGrouped;
  fragment.numSplitGroups = 1;
  fragment.groupedExecutionLeafNodeIds = {scanId};
  std::vector<std::unique_ptr<DriverFactory>> factories;
  core::QueryConfig config{{}};
  VELOX_ASSERT_THROW(
      LocalPlanner::plan(fragment, nullptr, &factories, config, 64),
      "Shared results do not support grouped execution");
  EXPECT_TRUE(factories.empty()); // Rejected before creating any pipelines.

  // The same CTE is accepted for ungrouped execution.
  fragment.executionStrategy = core::ExecutionStrategy::kUngrouped;
  fragment.numSplitGroups = 0;
  fragment.groupedExecutionLeafNodeIds.clear();
  EXPECT_NO_THROW(
      LocalPlanner::plan(fragment, nullptr, &factories, config, 64));
  EXPECT_FALSE(factories.empty());

  // Centralized CTE validation must not reject ordinary grouped fragments.
  auto ordinary = PlanBuilder().tableScan(ROW({"c0"}, {BIGINT()})).planNode();
  fragment = core::PlanFragment{ordinary};
  fragment.executionStrategy = core::ExecutionStrategy::kGrouped;
  fragment.numSplitGroups = 1;
  fragment.groupedExecutionLeafNodeIds = {ordinary->id()};
  factories.clear();
  EXPECT_NO_THROW(
      LocalPlanner::plan(fragment, nullptr, &factories, config, 64));
  EXPECT_FALSE(factories.empty());
}

TEST_F(CteTest, producerSortSpillsWithJoinSpillingEnabled) {
  auto batch =
      makeRowVector({makeFlatVector<int64_t>(4096, [](auto i) { return i; })});
  PlanBuilder query;
  core::PlanNodeId sortId;
  query.with("sorted", [&](PlanBuilder& cte) {
    cte.values(std::vector<RowVectorPtr>(16, batch))
        .orderBy({"c0 DESC"}, false)
        .capturePlanNodeId(sortId);
  });
  auto plan = query.fromCte("sorted").build();
  auto spillDirectory = common::testutil::TempDirectoryPath::create();
  TestScopedSpillInjection spill(100);
  auto task = AssertQueryBuilder(plan)
                  .spillDirectory(spillDirectory->getPath())
                  .config(core::QueryConfig::kSpillEnabled, true)
                  .config(core::QueryConfig::kOrderBySpillEnabled, true)
                  .config(core::QueryConfig::kJoinSpillEnabled, true)
                  .assertResults(std::vector<RowVectorPtr>(16, batch));
  EXPECT_GT(toPlanStats(task->taskStats()).at(sortId).spilledBytes, 0);
}

TEST_F(CteTest, hashJoinSpillingInProducersAndConsumers) {
  auto input = makeRowVector({makeFlatVector<int64_t>(
      4096, [](auto i) { return i; }, [](auto i) { return i % 97 == 0; })});
  createDuckDbTable("cte_input", {input});
  const std::vector<std::pair<core::JoinType, std::string>> joins{
      {core::JoinType::kInner, "INNER JOIN"},
      {core::JoinType::kLeft, "LEFT JOIN"},
      {core::JoinType::kRight, "RIGHT JOIN"},
      {core::JoinType::kFull, "FULL JOIN"},
      {core::JoinType::kLeftSemiFilter, "SEMI JOIN"},
      {core::JoinType::kAnti, "ANTI JOIN"}};
  for (bool producerJoin : {false, true}) {
    for (const auto& [joinType, sqlJoin] : joins) {
      const auto reference = fmt::format(
          "SELECT t.c0 FROM cte_input t {} (SELECT c0 * 2 AS r FROM cte_input) u ON t.c0 = u.r",
          sqlJoin);
      PlanBuilder query;
      core::PlanNodeId joinId;
      if (producerJoin) {
        query.with("joined", [&](PlanBuilder& cte) {
          auto right = cte.newPlan()
                           .values({input})
                           .project({"c0 * 2 as r"})
                           .localPartition({"r"})
                           .planNode();
          cte.values({input})
              .localPartition({"c0"})
              .hashJoin({"c0"}, {"r"}, right, "", {"c0"}, joinType)
              .capturePlanNodeId(joinId);
        });
        auto first = query.newPlan().fromCte("joined").planNode();
        auto second = query.newPlan().fromCte("joined").planNode();
        query.localPartition({}, {first, second});
      } else {
        query.with("base", [&](PlanBuilder& cte) {
          cte.values({input}).localPartition({"c0"});
        });
        auto right = query.newPlan()
                         .fromCte("base")
                         .project({"c0 * 2 as r"})
                         .localPartition({"r"})
                         .planNode();
        query.fromCte("base")
            .localPartition({"c0"})
            .hashJoin({"c0"}, {"r"}, right, "", {"c0"}, joinType)
            .capturePlanNodeId(joinId);
      }
      // Exercise normal serialized plan execution with no CTE-specific join
      // configuration, input copying, or spill implementation.
      auto plan = ISerializable::deserialize<core::PlanNode>(
          folly::parseJson(folly::toJson(query.build()->serialize())), pool());
      for (int drivers : {1, 64}) {
        for (bool spill : {false, true}) {
          SCOPED_TRACE(fmt::format(
              "producerJoin={} type={} drivers={} spill={}",
              producerJoin,
              sqlJoin,
              drivers,
              spill));
          auto directory = common::testutil::TempDirectoryPath::create();
          TestScopedSpillInjection injection(spill ? 100 : 0);
          auto task =
              AssertQueryBuilder(plan, duckDbQueryRunner_)
                  .maxDrivers(drivers)
                  .spillDirectory(directory->getPath())
                  .config(core::QueryConfig::kSpillEnabled, true)
                  .config(core::QueryConfig::kJoinSpillEnabled, true)
                  .config(core::QueryConfig::kMaxSpillLevel, 2)
                  .config(core::QueryConfig::kSpillWriteBufferSize, 0)
                  .assertResults(
                      producerJoin ? reference + " UNION ALL " + reference
                                   : reference);
          if (spill) {
            EXPECT_GT(
                toPlanStats(task->taskStats()).at(joinId).spilledBytes, 0);
          }
          EXPECT_TRUE(waitForTaskDriversToFinish(task.get()));
        }
      }
    }
  }
}

TEST_F(CteTest, spillingJoinProducerFeedsSpillingSelfJoin) {
  auto input = makeRowVector(
      {makeFlatVector<int64_t>(4096, [](auto i) { return i; }),
       makeFlatVector<std::string>(4096, [](auto i) {
         return fmt::format("long string payload from spilled build row {}", i);
       })});
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  PlanBuilder query(ids);
  core::PlanNodeId producerJoinId;
  core::PlanNodeId consumerJoinId;
  query.with("joined", [&](PlanBuilder& cte) {
    auto right = cte.newPlan()
                     .values({input})
                     .project({"c0 as r", "c1"})
                     .localPartition({"r"})
                     .planNode();
    cte.values({input})
        .project({"c0"})
        .localPartition({"c0"})
        .hashJoin({"c0"}, {"r"}, right, "", {"c0", "c1"})
        .capturePlanNodeId(producerJoinId);
  });
  auto right = query.newPlan()
                   .fromCte("joined")
                   .project({"c0 as r"})
                   .localPartition({"r"})
                   .planNode();
  auto plan = query.fromCte("joined")
                  .localPartition({"c0"})
                  .hashJoin({"c0"}, {"r"}, right, "", {"c0", "c1"})
                  .capturePlanNodeId(consumerJoinId)
                  .build();
  for (int drivers : {1, 64}) {
    for (bool earlyLimit : {false, true}) {
      SCOPED_TRACE(
          fmt::format("drivers={} earlyLimit={}", drivers, earlyLimit));
      auto limited = PlanBuilder(plan, ids)
                         .project({"cast(1 as bigint) as c0"})
                         .limit(0, 7, false)
                         .planNode();
      auto expected = earlyLimit
          ? makeRowVector({makeFlatVector<int64_t>(7, [](auto) { return 1; })})
          : input;
      auto directory = common::testutil::TempDirectoryPath::create();
      TestScopedSpillInjection injection(100);
      auto task = AssertQueryBuilder(earlyLimit ? limited : plan)
                      .maxDrivers(drivers)
                      .spillDirectory(directory->getPath())
                      .config(core::QueryConfig::kSpillEnabled, true)
                      .config(core::QueryConfig::kJoinSpillEnabled, true)
                      .config(core::QueryConfig::kMaxSpillLevel, 2)
                      .config(core::QueryConfig::kSpillWriteBufferSize, 0)
                      .assertResults(expected);
      const auto stats = toPlanStats(task->taskStats());
      EXPECT_GT(stats.at(producerJoinId).spilledBytes, 0);
      EXPECT_GT(stats.at(consumerJoinId).spilledBytes, 0);
      EXPECT_TRUE(waitForTaskDriversToFinish(task.get()));
    }
  }
}

TEST_F(CteTest, taskCancellationWakesCteWaiters) {
  auto input =
      makeRowVector({makeFlatVector<int64_t>(128, [](auto i) { return i; })});
  PlanBuilder query;
  query.with("t", [&](PlanBuilder& cte) {
    cte.tableScan(input->rowType()); // No splits: all readers wait.
  });
  auto first = query.newPlan().fromCte("t").planNode();
  auto second = query.newPlan().fromCte("t").planNode();
  query.localPartition({}, {first, second});
  auto plan = query.build();
  CursorParameters params;
  params.planNode = plan;
  params.maxDrivers = 64;
  params.bufferedBytes = 1;
  params.queryCtx =
      core::QueryCtx::Builder().executor(driverExecutor_.get()).build();
  params.queryCtx->testingOverrideConfigUnsafe(
      {{core::QueryConfig::kMaxLocalExchangeBufferSize, "1"}});
  auto cursor = TaskCursor::create(params);
  auto task = cursor->task();
  SCOPE_EXIT {
    if (task->isRunning()) {
      task->requestCancel().wait();
    }
  };
  cursor->start();
  ASSERT_TRUE(waitUntil([&]() {
    auto stats = task->taskStats();
    if (stats.numRunningDrivers != 0 || stats.numQueuedDrivers != 0) {
      return false;
    }
    return stats.numBlockedDrivers[BlockingReason::kWaitForProducer] >= 128;
  }));
  auto buffer = std::dynamic_pointer_cast<detail::SharedResultBuffer>(
      task->getCustomJoinBridge(kUngroupedGroupId, plan->id()));
  ASSERT_NE(buffer, nullptr);
  ContinueFuture observer = ContinueFuture::makeEmpty();
  EXPECT_TRUE(
      buffer->readerBlocked(buffer->readerIndex(first->id()), &observer));
  ASSERT_TRUE(observer.valid());
  EXPECT_FALSE(observer.isReady());
  auto cancelled = task->requestCancel();
  EXPECT_TRUE(observer.isReady());
  EXPECT_TRUE(std::move(cancelled).wait(std::chrono::seconds(5)));
  EXPECT_TRUE(waitForTaskCancelled(task.get()));
  EXPECT_TRUE(waitForTaskDriversToFinish(task.get()));
  VELOX_ASSERT_THROW(
      buffer->readerBlocked(buffer->readerIndex(first->id()), &observer),
      "Shared result cancelled");
}

TEST_F(CteTest, zeroCopyReadAndArbitrationOnWrite) {
  auto input =
      makeRowVector({makeFlatVector<int64_t>(1024, [](auto i) { return i; })});
  auto buffer = makeBuffer(2, {1 << 20});
  buffer->add(
      std::static_pointer_cast<RowVector>(BaseVector::copy(*input, pool())));
  buffer->finish();

  // Reading should require no payload allocation. Only a downstream write
  // should allocate, triggering real arbitration in this zero-capacity pool.
  memory::MemoryManager::Options options;
  options.allocatorCapacity = 64 << 20;
  options.arbitratorCapacity = 64 << 20;
  options.arbitratorKind = "SHARED";
  using ExtraConfig = memory::SharedArbitrator::ExtraConfig;
  options.extraArbitratorConfigs = {
      {std::string(ExtraConfig::kMemoryPoolInitialCapacity), "0B"},
      {std::string(ExtraConfig::kMemoryPoolReservedCapacity), "0B"}};
  int arbitrations = 0;
  options.arbitrationStateCheckCb = [&](memory::MemoryPool&) {
    ++arbitrations;
    // Consumer mutation and arbitration must happen outside the queue lock.
    EXPECT_FALSE(buffer->allReadersClosed());
    EXPECT_TRUE(buffer->atEnd(0)); // The last reader has already claimed it.
  };
  memory::MemoryManager manager(options);
  auto root = manager.addRootPool("cte-copy");
  auto copyPool = root->addLeafChild("reader");
  ASSERT_EQ(root->capacity(), 0);
  auto output = buffer->next(0, copyPool.get());
  EXPECT_EQ(arbitrations, 0);
  EXPECT_EQ(root->capacity(), 0);
  output = writable(std::move(output), copyPool.get());
  EXPECT_GT(arbitrations, 0);
  assertEqualResults({input}, {output});
  buffer->closeReader(0);
  buffer->closeReader(1);
}

TEST_F(CteTest, sharedPlanAndEarlyLimit) {
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto batch =
      makeRowVector({makeFlatVector<int64_t>(100, [](auto i) { return i; })});
  std::vector<RowVectorPtr> input(20, batch);
  Cte cte(ids->next(), PlanBuilder(ids).values(input).planNode());
  auto limited =
      PlanBuilder(cte.scan(ids->next()), ids).limit(0, 7, false).planNode();
  auto all = PlanBuilder(cte.scan(ids->next()), ids).project({"c0"}).planNode();
  auto query = PlanBuilder(ids).localPartition({}, {limited, all}).planNode();
  auto plan = finalizeCtes(query, {1 << 20});
  EXPECT_NE(plan->toString(true, true).find("2 readers"), std::string::npos);
  auto expected = input;
  expected.push_back(
      makeRowVector({makeFlatVector<int64_t>(7, [](auto i) { return i; })}));
  for (int run = 0; run < 3; ++run) {
    auto task = AssertQueryBuilder(plan).maxDrivers(4).assertResults(expected);
    EXPECT_EQ(positions(*task, "Values", false), 2000);
    EXPECT_EQ(positions(*task, "SharedResultSink", true), 2000);
  }
}

TEST_F(CteTest, q15RevenueAndMaximum) {
  auto input = makeRowVector(
      {makeFlatVector<int64_t>(6000, [](auto i) { return i; }),
       makeFlatVector<int64_t>(6000, [](auto i) { return i % 13; })});
  PlanBuilder query;
  query.with("revenue", [&](PlanBuilder& cte) {
    cte.values({input}).singleAggregation({"c0"}, {"sum(c1) as revenue"});
  });
  auto maximum = query.newPlan()
                     .fromCte("revenue")
                     .partialAggregation({}, {"max(revenue) as max_revenue"})
                     .localPartition({})
                     .finalAggregation()
                     .planNode();
  query.fromCte("revenue").hashJoin(
      {"revenue"}, {"max_revenue"}, maximum, "", {"c0", "revenue"});
  auto plan = query.build({1 << 20});
  EXPECT_NE(plan->toString(true, true).find("2 readers"), std::string::npos);
  std::vector<int64_t> suppliers;
  for (int64_t i = 0; i < 6000; ++i) {
    if (i % 13 == 12) {
      suppliers.push_back(i);
    }
  }
  auto expected = makeRowVector(
      {makeFlatVector(suppliers),
       makeFlatVector<int64_t>(suppliers.size(), [](auto) { return 12; })});
  // The producer can finish while the probe waits for its own build branch
  // to consume the same CTE.
  for (int run = 0; run < 3; ++run) {
    auto task = AssertQueryBuilder(plan).maxDrivers(4).assertResults(expected);
    EXPECT_EQ(positions(*task, "Values", false), 6000);
    EXPECT_EQ(positions(*task, "SharedResultSink", true), 6000);
    EXPECT_EQ(positions(*task, "SharedScan", false), 12000);
  }
  auto tooSmall = query.build({1});
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(tooSmall).copyResults(pool()),
      "Shared result retained-byte limit exceeded");
}

TEST_F(CteTest, emptySharedInput) {
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto empty = makeRowVector({makeFlatVector<int64_t>({})});
  Cte cte(ids->next(), PlanBuilder(ids).values({empty}).planNode());
  auto query =
      PlanBuilder(ids)
          .localPartition({}, {cte.scan(ids->next()), cte.scan(ids->next())})
          .planNode();
  AssertQueryBuilder(finalizeCtes(query)).assertEmptyResults();
}

TEST_F(CteTest, scanSplitsAreConsumedOnce) {
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto batch = makeRowVector(
      {makeFlatVector<int64_t>(1000, [](auto i) { return i; }),
       makeFlatVector<std::string>(1000, [](auto i) {
         return fmt::format("a long string requiring external storage: {}", i);
       })});
  auto file = writeToFile(batch);
  auto scan = PlanBuilder(ids).tableScan(asRowType(batch->type())).planNode();
  Cte cte(ids->next(), scan);
  auto query =
      PlanBuilder(ids)
          .localPartition({}, {cte.scan(ids->next()), cte.scan(ids->next())})
          .planNode();
  auto task =
      AssertQueryBuilder(finalizeCtes(query, {1 << 20}))
          .maxDrivers(4)
          .split(scan->id(), makeHiveConnectorSplit(file->getPath()))
          .split(scan->id(), makeHiveConnectorSplit(file->getPath()))
          .assertResults(std::vector<RowVectorPtr>{batch, batch, batch, batch});
  EXPECT_EQ(positions(*task, "TableScan", false), 2000);
  EXPECT_EQ(positions(*task, "SharedResultSink", true), 2000);
  EXPECT_EQ(positions(*task, "SharedScan", false), 4000);
}

TEST_F(CteTest, parallelScanAt64Drivers) {
  constexpr int kDrivers = 64;
  constexpr int kSplits = 256;
  constexpr int kRows = 256;
  std::vector<RowVectorPtr> batches;
  auto files = makeFilePaths(kSplits);
  for (int split = 0; split < kSplits; ++split) {
    batches.push_back(makeRowVector(
        {makeFlatVector<int64_t>(
             kRows, [split](auto i) { return split * kRows + i; }),
         makeFlatVector<std::string>(kRows, [split](auto i) {
           return fmt::format(
               "unique split {} row {} externally stored payload", split, i);
         })}));
    writeToFile(files[split]->getPath(), batches.back());
  }
  PlanBuilder query;
  core::PlanNodeId scanId;
  query.with("source", [&](PlanBuilder& cte) {
    cte.tableScan(batches[0]->rowType()).capturePlanNodeId(scanId);
  });
  auto branch = query.newPlan();
  auto left = branch.newPlan().fromCte("source").planNode();
  auto right = branch.newPlan().fromCte("source").planNode();
  branch.localPartition({}, {left, right});
  auto plan = branch.build();
  AssertQueryBuilder assertion(plan);
  assertion.maxDrivers(kDrivers);
  for (int i = 0; i < kSplits; ++i) {
    assertion.split(scanId, makeHiveConnectorSplit(files[i]->getPath()));
  }
  auto expected = batches;
  expected.insert(expected.end(), batches.begin(), batches.end());
  auto task = assertion.assertResults(expected);
  EXPECT_EQ(positions(*task, "TableScan", false), kSplits * kRows);
  EXPECT_EQ(positions(*task, "SharedResultSink", true), kSplits * kRows);
  EXPECT_EQ(positions(*task, "SharedScan", false), 2 * kSplits * kRows);
  int scanOperators = 0;
  for (const auto& pipeline : task->taskStats().pipelineStats) {
    for (const auto& op : pipeline.operatorStats) {
      if (op.operatorType == "SharedResultSink" ||
          op.operatorType == "SharedScan") {
        EXPECT_EQ(op.numDrivers, kDrivers);
        const auto metric = op.operatorType == "SharedScan"
            ? "activeReaderDrivers"
            : "activeProducerDrivers";
        EXPECT_GT(op.runtimeStats.at(metric).sum, 1);
        if (op.operatorType == "SharedScan") {
          ++scanOperators;
        }
      }
    }
  }
  EXPECT_EQ(scanOperators, 2);
}

TEST_F(CteTest, parallelGroupedProducerAndDependentJoin) {
  constexpr int kDrivers = 64;
  constexpr int kSplits = 64;
  constexpr int kRows = 1024;
  constexpr int kKeys = 64;
  auto batch = makeRowVector(
      {makeFlatVector<int64_t>(kRows, [](auto i) { return i % kKeys; }),
       makeFlatVector<int64_t>(kRows, [](auto i) { return i % 13; })});
  auto file = writeToFile(batch);
  PlanBuilder query;
  core::PlanNodeId scanId;
  query.with("revenue", [&](PlanBuilder& cte) {
    cte.tableScan(asRowType(batch->type()))
        .capturePlanNodeId(scanId)
        .partialAggregation({"c0"}, {"sum(c1) as revenue"})
        .localPartition({"c0"})
        .finalAggregation();
  });
  auto maximum = query.newPlan()
                     .fromCte("revenue")
                     .partialAggregation({}, {"max(revenue) as maximum"})
                     .localPartition({})
                     .finalAggregation()
                     .planNode();
  auto plan =
      query.fromCte("revenue")
          .hashJoin({"revenue"}, {"maximum"}, maximum, "", {"c0", "revenue"})
          .build();
  std::vector<int64_t> totals(kKeys);
  for (int i = 0; i < kRows; ++i) {
    totals[i % kKeys] += (i % 13) * kSplits;
  }
  const auto max = *std::max_element(totals.begin(), totals.end());
  std::vector<int64_t> keys;
  for (int i = 0; i < kKeys; ++i) {
    if (totals[i] == max) {
      keys.push_back(i);
    }
  }
  auto expected = makeRowVector(
      {makeFlatVector(keys),
       makeFlatVector<int64_t>(keys.size(), [max](auto) { return max; })});
  AssertQueryBuilder assertion(plan);
  assertion.maxDrivers(kDrivers);
  for (int i = 0; i < kSplits; ++i) {
    assertion.split(scanId, makeHiveConnectorSplit(file->getPath()));
  }
  auto task = assertion.assertResults(expected);
  EXPECT_EQ(positions(*task, "TableScan", false), kRows * kSplits);
  EXPECT_EQ(positions(*task, "SharedResultSink", true), kKeys);
  EXPECT_EQ(positions(*task, "SharedScan", false), 2 * kKeys);
  for (const auto& pipeline : task->taskStats().pipelineStats) {
    for (const auto& op : pipeline.operatorStats) {
      if (op.operatorType == "SharedResultSink" ||
          op.operatorType == "SharedScan") {
        EXPECT_EQ(op.numDrivers, kDrivers);
      }
    }
  }
}

// Manual scaling experiment, excluded from normal correctness runs. Run with
// --gtest_also_run_disabled_tests
// --gtest_filter=CteTest.DISABLED_parallelScaling
TEST_F(CteTest, DISABLED_parallelScaling) {
  const bool sha = std::getenv("VELOX_CTE_SCALING_SHA") != nullptr;
  constexpr int kRows = 8192;
  const int kSplits = sha ? 64 : 1024;
  auto batch = makeRowVector(
      {makeFlatVector<int64_t>(kRows, [](auto i) { return i; }),
       makeFlatVector<std::string>(kRows, [](auto i) {
         return std::string(1024, 'x') + std::to_string(i);
       })});
  auto file = writeToFile(batch);
  PlanBuilder query;
  core::PlanNodeId scanId;
  query.with("computed", [&](PlanBuilder& cte) {
    cte.tableScan(asRowType(batch->type()))
        .capturePlanNodeId(scanId)
        .project(
            {"c0",
             sha ? "sha256(to_utf8(c1)) as value"
                 : "sin(sin(sin(sin(cast(c0 as double))))) as value"});
  });
  auto consume = [&]() {
    auto branch = query.newPlan().fromCte("computed");
    std::string expression = "value";
    for (int i = 0; i < (sha ? 8 : 64); ++i) {
      expression = std::string(sha ? "sha256(" : "sin(") + expression + ")";
    }
    return branch
        .project(
            {"c0",
             (sha ? "length(" + expression + ")"
                  : "if(" + expression + " > 0.0, BIGINT '1', BIGINT '0')") +
                 " as result"})
        .partialAggregation(
            {},
            {"count(*) as n",
             "sum(c0) as id_sum",
             "sum(result) as computed_sum"})
        .localPartition({})
        .finalAggregation()
        .planNode();
  };
  auto first = consume();
  auto second = consume();
  auto plan = query.localPartition({}, {first, second}).build({256 << 20});
  int64_t computedSum = sha ? int64_t(kRows) * 32 : 0;
  if (!sha) {
    for (int i = 0; i < kRows; ++i) {
      computedSum += std::sin(double(i)) > 0;
    }
  }
  auto expected = makeRowVector(
      {makeFlatVector<int64_t>({kRows * kSplits, kRows * kSplits}),
       makeFlatVector<int64_t>(
           {int64_t(kRows) * (kRows - 1) / 2 * kSplits,
            int64_t(kRows) * (kRows - 1) / 2 * kSplits}),
       makeFlatVector<int64_t>(
           {computedSum * kSplits, computedSum * kSplits})});
  for (int drivers : {1, 8, 32, 64}) {
    std::vector<double> samples;
    for (int run = 0; run < 4; ++run) {
      AssertQueryBuilder assertion(plan);
      assertion.maxDrivers(drivers);
      for (int i = 0; i < kSplits; ++i) {
        assertion.split(scanId, makeHiveConnectorSplit(file->getPath()));
      }
      auto start = std::chrono::steady_clock::now();
      auto task = assertion.assertResults(expected);
      double seconds = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - start)
                           .count();
      const auto stats = task->taskStats();
      double taskSeconds =
          (stats.executionEndTimeMs - stats.executionStartTimeMs) / 1000.0;
      if (run != 0) { // Warm each configuration before recording timings.
        samples.push_back(taskSeconds);
      }
      int64_t cpuNanos = 0;
      int64_t producers = 0;
      int64_t readers = 0;
      for (const auto& pipeline : stats.pipelineStats) {
        for (const auto& op : pipeline.operatorStats) {
          const auto operatorCpu = op.addInputTiming.cpuNanos +
              op.getOutputTiming.cpuNanos + op.finishTiming.cpuNanos +
              op.isBlockedTiming.cpuNanos;
          cpuNanos += operatorCpu;
          if (run == 1) {
            std::cout << "CTE_OPERATOR_CPU drivers=" << drivers
                      << " node=" << op.planNodeId
                      << " operator=" << op.operatorType
                      << " seconds=" << operatorCpu / 1e9 << std::endl;
          }
          if (auto it = op.runtimeStats.find("activeProducerDrivers");
              it != op.runtimeStats.end()) {
            producers += it->second.sum;
          }
          if (auto it = op.runtimeStats.find("activeReaderDrivers");
              it != op.runtimeStats.end()) {
            readers += it->second.sum;
          }
        }
      }
      std::cout << "CTE_SCALING workload=" << (sha ? "sha" : "math")
                << " drivers=" << drivers << " run=" << run
                << " seconds=" << seconds << " task_seconds=" << taskSeconds
                << " cpu_seconds=" << cpuNanos / 1e9
                << " active_producers=" << producers
                << " active_readers=" << readers << std::endl;
    }
    std::sort(samples.begin(), samples.end());
    std::cout << "CTE_SCALING_MEDIAN drivers=" << drivers
              << " seconds=" << samples[1] << std::endl;
  }
}

TEST_F(CteTest, sharedVolatileExpressionIsEvaluatedOnce) {
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto input =
      makeRowVector({makeFlatVector<int64_t>(100, [](auto i) { return i; })});
  auto shared = PlanBuilder(ids)
                    .values({input})
                    .project({"c0", "random() as r"})
                    .planNode();
  Cte cte(ids->next(), shared);
  auto right = PlanBuilder(cte.scan(ids->next()), ids)
                   .project({"c0 as right_id", "r as right_r"})
                   .planNode();
  auto dag =
      PlanBuilder(cte.scan(ids->next()), ids)
          .hashJoin({"c0", "r"}, {"right_id", "right_r"}, right, "", {"c0"})
          .planNode();
  // Every ID matches because both readers see the same random value, even
  // though the producer contains a non-deterministic expression.
  AssertQueryBuilder(finalizeCtes(dag, {1 << 20}))
      .maxDrivers(64)
      .assertResults(input);
}

TEST_F(CteTest, builderCteScopeAndFinalization) {
  auto batch = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  PlanBuilder query;
  auto beforeDefinition = query.newPlan();
  query.with("numbers", [&](PlanBuilder& cte) { cte.values({batch}); });
  VELOX_ASSERT_THROW(beforeDefinition.fromCte("numbers"), "Unknown CTE");
  auto branch = query.newPlan();
  branch.with("local", [](PlanBuilder& cte) { cte.fromCte("numbers"); });
  VELOX_ASSERT_THROW(query.fromCte("local"), "Unknown CTE");
  branch.fromCte("local");
  // Explicit definitions keep their semantics even with one use. Finalization
  // is repeatable without mutating the logical plan or definition.
  EXPECT_NE(branch.build(), branch.planNode());
  EXPECT_EQ(
      branch.build()->toString(true, true),
      branch.build()->toString(true, true));
  AssertQueryBuilder(branch.build()).assertResults(batch);
  VELOX_ASSERT_THROW(branch.fromCte("numbers"), "empty builder");
  VELOX_ASSERT_THROW(query.build(), "empty plan");
  VELOX_ASSERT_THROW(query.with("", [](PlanBuilder&) {}), "must not be empty");
  VELOX_ASSERT_THROW(
      query.with("numbers", [](PlanBuilder&) {}), "already defined");
  VELOX_ASSERT_THROW(query.with("empty", [](PlanBuilder&) {}), "has no plan");
  VELOX_ASSERT_THROW(
      query.with(
          "recursive", [](PlanBuilder& cte) { cte.fromCte("recursive"); }),
      "Unknown CTE");
  // Failed definitions leave no binding behind.
  query.with("empty", [](PlanBuilder& cte) { cte.fromCte("numbers"); });
  query.fromCte("empty");
  AssertQueryBuilder(query.build()).assertResults(batch);
}

TEST_F(CteTest, arbitraryProducerAndConsumerOperators) {
  auto input = makeRowVector(
      {makeFlatVector<int64_t>(6000, [](auto i) { return i; }),
       makeFlatVector<int64_t>(6000, [](auto i) { return i % 13; })});
  auto offset = makeRowVector({"delta"}, {makeFlatVector<int64_t>({100})});
  PlanBuilder query;
  core::PlanNodePtr producer;
  query.with("ranked", [&](PlanBuilder& cte) {
    auto right = cte.newPlan().values({offset}).planNode();
    cte.values({input})
        .localPartition({"c1"})
        .window({"row_number() over (partition by c1 order by c0) as rn"})
        .filter("rn = 1")
        .nestedLoopJoin(right, {"c0", "delta"})
        .project({"c0 + delta as n"})
        .topN({"n"}, 10, false);
    producer = cte.planNode();
  });
  auto maximum = query.newPlan()
                     .fromCte("ranked")
                     .partialAggregation({}, {"max(n) as maximum"})
                     .localPartition({})
                     .finalAggregation()
                     .planNode();
  query.fromCte("ranked")
      .localPartition({"n"})
      .nestedLoopJoin(maximum, "n = maximum", {"n"})
      .topN({"n DESC"}, 1, false);
  auto plan = query.build({1 << 20});
  // Neither subtree is cloned or rewritten, regardless of its operator types.
  EXPECT_EQ(plan->sources()[0], query.planNode());
  EXPECT_EQ(plan->sources()[1], producer);
  auto expected = makeRowVector({makeFlatVector<int64_t>({109})});
  for (int drivers : {1, 4, 64}) {
    auto task =
        AssertQueryBuilder(plan).maxDrivers(drivers).assertResults(expected);
    EXPECT_EQ(positions(*task, "SharedResultSink", true), 10);
    EXPECT_EQ(positions(*task, "Values", false), 6001);
  }
}

TEST_F(CteTest, multipleDependentAndUnusedDefinitions) {
  auto batch = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  PlanBuilder query;
  query.with("base", [&](PlanBuilder& cte) { cte.values({batch}); });
  query.with("unused", [&](PlanBuilder& cte) {
    cte.values({batch}).project({"c0 / 0 as c0"});
  });
  query.with("dependent", [](PlanBuilder& cte) {
    cte.with("local", [](PlanBuilder& nested) {
      nested.fromCte("base").project({"c0 + 10 as c0"});
    });
    cte.fromCte("local");
  });
  query.with("independent", [&](PlanBuilder& cte) {
    cte.values({batch}).project({"c0 + 100 as c0"});
  });
  // An abandoned reference must not pin buffered data or create an extra
  // reader.
  auto abandoned = query.newPlan().fromCte("base").planNode();
  auto base = query.newPlan().fromCte("base").planNode();
  auto dependent = query.newPlan().fromCte("dependent").planNode();
  auto independent = query.newPlan().fromCte("independent").planNode();
  query.localPartition({}, {base, dependent, independent});
  auto plan = query.build();
  auto expected = makeRowVector(
      {makeFlatVector<int64_t>({1, 2, 3, 11, 12, 13, 101, 102, 103})});
  for (int drivers : {1, 4, 64}) {
    auto task =
        AssertQueryBuilder(plan).maxDrivers(drivers).assertResults(expected);
    EXPECT_EQ(positions(*task, "Values", false), 6);
    // base, local, dependent, independent are each produced exactly once.
    EXPECT_EQ(positions(*task, "SharedResultSink", true), 12);
    EXPECT_EQ(positions(*task, "SharedScan", false), 15);
  }
  EXPECT_EQ(core::PlanNode::findNodeById(plan.get(), abandoned->id()), nullptr);
}

TEST_F(CteTest, serialDependentSelfJoin) {
  auto input =
      makeRowVector({makeFlatVector<int64_t>(256, [](auto i) { return i; })});
  PlanBuilder query;
  query.with("base", [&](PlanBuilder& cte) { cte.values({input}); });
  query.with("derived", [&](PlanBuilder& cte) {
    cte.fromCte("base").project({"c0 + 1 as c0"});
  });
  auto right =
      query.newPlan().fromCte("derived").project({"c0 as r"}).planNode();
  auto plan = query.fromCte("derived")
                  .hashJoin({"c0"}, {"r"}, right, "", {"c0"})
                  .build();
  auto expected = makeRowVector(
      {makeFlatVector<int64_t>(256, [](auto i) { return i + 1; })});
  auto task =
      AssertQueryBuilder(plan).serialExecution(true).assertResults(expected);
  EXPECT_EQ(positions(*task, "Values", false), 256);
  EXPECT_EQ(positions(*task, "SharedResultSink", true), 512);
  EXPECT_EQ(positions(*task, "SharedScan", false), 768);
  EXPECT_TRUE(waitForTaskDriversToFinish(task.get()));
}

TEST_F(CteTest, allReadersEarlyLimit) {
  auto input =
      makeRowVector({makeFlatVector<int64_t>(128, [](auto) { return 1; })});
  auto expected =
      makeRowVector({makeFlatVector<int64_t>(18, [](auto) { return 1; })});
  for (bool serial : {false, true}) {
    for (int drivers : {1, 64}) {
      if (serial && drivers != 1) {
        continue;
      }
      PlanBuilder query;
      query.with(
          "base", [&](PlanBuilder& cte) { cte.values({input}, false, 10000); });
      auto left = query.newPlan().fromCte("base").planNode();
      auto right = query.newPlan().fromCte("base").planNode();
      auto leftLimited = query.newPlan()
                             .addNode([&](auto, auto) { return left; })
                             .limit(0, 7, false)
                             .planNode();
      auto rightLimited = query.newPlan()
                              .addNode([&](auto, auto) { return right; })
                              .limit(0, 11, false)
                              .planNode();
      query.localPartition({}, {leftLimited, rightLimited});
      auto plan = query.build();
      auto task = AssertQueryBuilder(plan)
                      .serialExecution(serial)
                      .maxDrivers(drivers)
                      .assertResults(expected);
      EXPECT_TRUE(waitForTaskDriversToFinish(task.get()));
    }
  }
}

TEST_F(CteTest, allReadersCloseWhileProducerAwaitsSplits) {
  auto input =
      makeRowVector({makeFlatVector<int64_t>(128, [](auto) { return 1; })});
  auto file = writeToFile(input);
  for (int drivers : {1, 64}) {
    PlanBuilder query;
    core::PlanNodeId scanId;
    query.with("base", [&](PlanBuilder& cte) {
      cte.tableScan(input->rowType()).capturePlanNodeId(scanId);
    });
    auto left = query.newPlan().fromCte("base").limit(0, 7, false).planNode();
    auto right = query.newPlan().fromCte("base").limit(0, 11, false).planNode();
    auto plan = query.localPartition({}, {left, right}).build();
    CursorParameters params;
    params.planNode = plan;
    params.maxDrivers = drivers;
    params.queryCtx =
        core::QueryCtx::Builder().executor(driverExecutor_.get()).build();
    auto cursor = TaskCursor::create(params);
    auto task = cursor->task();
    SCOPE_EXIT {
      if (task->isRunning()) {
        task->requestCancel().wait();
      }
    };
    task->addSplit(scanId, Split(makeHiveConnectorSplit(file->getPath())));
    // Deliberately do not call noMoreSplits(): early reader closure must finish
    // the task.
    int rows = 0;
    while (cursor->moveNext()) {
      rows += cursor->current()->size();
    }
    EXPECT_EQ(rows, 18);
    EXPECT_TRUE(waitForTaskCompletion(task.get()));
    EXPECT_TRUE(waitForTaskDriversToFinish(task.get()));
  }
}

TEST_F(CteTest, serializationPreservesSharingAndDependencies) {
  auto input = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  auto doubled = makeRowVector({makeFlatVector<int64_t>({2, 4, 6})});
  PlanBuilder query;
  query.with("base", [&](PlanBuilder& cte) { cte.values({input}); });
  query.with("derived", [&](PlanBuilder& cte) {
    cte.fromCte("base").project({"c0 * 2 as c0"});
  });
  auto first = query.newPlan().fromCte("base").planNode();
  auto second = query.newPlan().fromCte("derived").planNode();
  auto third = query.newPlan().fromCte("derived").planNode();
  query.localPartition({}, {first, second, third});
  for (auto limit :
       {std::optional<uint64_t>{},
        std::optional<uint64_t>{1 << 20},
        std::optional<uint64_t>{std::numeric_limits<uint64_t>::max()}}) {
    const auto plan = query.build({limit});
    // Go through actual JSON, including the unlimited and full uint64 limits.
    const auto serialized = folly::parseJson(folly::toJson(plan->serialize()));
    const auto restored =
        ISerializable::deserialize<core::PlanNode>(serialized, pool());
    EXPECT_EQ(restored->serialize(), serialized);
    EXPECT_EQ(restored->toString(true, true), plan->toString(true, true));
    for (auto drivers : {1, 64}) {
      auto task =
          AssertQueryBuilder(restored).maxDrivers(drivers).assertResults(
              std::vector<RowVectorPtr>{input, doubled, doubled});
      // The shared base must execute once, not once per serialized reference.
      EXPECT_EQ(positions(*task, "Values", false), 3);
    }
  }
}

TEST_F(CteTest, serializationRejectsInvalidReferences) {
  auto input = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  PlanBuilder query;
  query.with("base", [&](PlanBuilder& cte) { cte.values({input}); });
  auto plan = query.fromCte("base").build();
  auto restore = [&](const folly::dynamic& obj) {
    return ISerializable::deserialize<core::PlanNode>(obj, pool());
  };
  auto obj = plan->serialize();
  obj["sources"][0]["sharedId"] = "missing";
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(restore(obj)).copyResults(pool()), "Unbound CTE");
  obj = plan->serialize();
  obj["readers"] = folly::dynamic::array("missing");
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(restore(obj)).copyResults(pool()),
      "reader IDs do not match scans");
  obj = plan->serialize();
  obj["sources"][0]["outputType"] = ROW({"c0"}, {VARCHAR()})->serialize();
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(restore(obj)).copyResults(pool()),
      "output type does not match producer");
  obj = plan->serialize();
  obj["sources"][1] = obj["sources"][0];
  obj["sources"][1]["id"] = "recursive-scan";
  obj["readers"].push_back("recursive-scan");
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(restore(obj)).copyResults(pool()), "Recursive CTE");
  obj = plan->serialize();
  obj["maxRetainedBytes"] = "0";
  VELOX_ASSERT_THROW(restore(obj), "maxRetainedBytes must be positive");
}

TEST_F(CteTest, queueMetadataUsesQueryMemoryArbitration) {
  auto empty = std::make_shared<RowVector>(
      pool(), ROW({}, {}), nullptr, 1, std::vector<VectorPtr>{});
  auto buffer = makeBuffer(2, {});
  memory::MemoryManager::Options options;
  options.allocatorCapacity = 8LL << 20;
  options.arbitratorCapacity = 8LL << 20;
  options.arbitratorKind = "SHARED";
  using ExtraConfig = memory::SharedArbitrator::ExtraConfig;
  options.extraArbitratorConfigs = {
      {std::string(ExtraConfig::kMemoryPoolInitialCapacity), "0B"},
      {std::string(ExtraConfig::kMemoryPoolReservedCapacity), "0B"}};
  int arbitrations = 0;
  options.arbitrationStateCheckCb = [&](memory::MemoryPool&) {
    ++arbitrations;
    // Reentering the queue during reservation must not deadlock.
    EXPECT_FALSE(buffer->allReadersClosed());
  };
  memory::MemoryManager manager(options);
  auto root = manager.addRootPool("cte-metadata", 1LL << 20);
  auto metadataPool = root->addLeafChild("sink");
  bool exceeded = false;
  for (int i = 0; i < 10000; ++i) {
    try {
      buffer->add(empty, metadataPool.get());
    } catch (const VeloxException& error) {
      EXPECT_EQ(error.errorCode(), error_code::kMemCapExceeded);
      exceeded = true;
      break;
    }
  }
  EXPECT_TRUE(exceeded);
  EXPECT_GT(arbitrations, 0);
  const auto retained = root->usedBytes();
  EXPECT_GT(retained, 0);
  buffer->next(0, pool());
  EXPECT_EQ(root->usedBytes(), retained);
  buffer->next(1, pool());
  EXPECT_LT(root->usedBytes(), retained);
  buffer->closeReader(0);
  buffer->closeReader(1);
  EXPECT_EQ(root->usedBytes(), 0);
}

TEST_F(CteTest, cancellationDuringMetadataArbitration) {
  auto empty = std::make_shared<RowVector>(
      pool(), ROW({}, {}), nullptr, 1, std::vector<VectorPtr>{});
  auto buffer = makeBuffer(1, {});
  memory::MemoryManager::Options options;
  options.allocatorCapacity = 8LL << 20;
  options.arbitratorCapacity = 8LL << 20;
  options.arbitratorKind = "SHARED";
  using ExtraConfig = memory::SharedArbitrator::ExtraConfig;
  options.extraArbitratorConfigs = {
      {std::string(ExtraConfig::kMemoryPoolInitialCapacity), "0B"},
      {std::string(ExtraConfig::kMemoryPoolReservedCapacity), "0B"}};
  options.arbitrationStateCheckCb = [&](memory::MemoryPool&) {
    buffer->cancel();
  };
  memory::MemoryManager manager(options);
  auto root = manager.addRootPool("cte-cancel-metadata");
  auto metadataPool = root->addLeafChild("sink");
  VELOX_ASSERT_THROW(
      buffer->add(empty, metadataPool.get()), "Shared result cancelled");
  EXPECT_EQ(root->usedBytes(), 0);
  EXPECT_EQ(metadataPool->stats().numExternalAllocs, 1);
  EXPECT_EQ(metadataPool->stats().numExternalFrees, 1);
}

TEST_F(CteTest, unlimitedRetentionUsesQueryMemoryArbitration) {
  // The join probe cannot consume until its build has read the entire CTE.
  // Producer projections allocate fresh query-owned buffers, unlike Values
  // inputs supplied by the application from an external pool.
  constexpr int32_t kRows = 8192;
  constexpr int32_t kBatches = 1280;
  auto input =
      makeRowVector({makeFlatVector<int64_t>(kRows, [](auto i) { return i; })});
  PlanBuilder query;
  query.with("base", [&](PlanBuilder& cte) {
    cte.values({input}, false, kBatches).project({"c0 + 1 as c0"});
  });
  auto maximum = query.newPlan()
                     .fromCte("base")
                     .singleAggregation({}, {"max(c0) as maximum"})
                     .planNode();
  auto plan = query.fromCte("base")
                  .hashJoin({"c0"}, {"maximum"}, maximum, "", {"c0"})
                  .singleAggregation({}, {"count(1) as n"})
                  .build();
  auto expected = makeRowVector({makeFlatVector<int64_t>({kBatches})});

  for (int64_t capacity : {8LL << 20, 256LL << 20}) {
    SCOPED_TRACE(capacity);
    memory::MemoryManager::Options options;
    options.allocatorCapacity = 512LL << 20;
    options.arbitratorCapacity = 512LL << 20;
    options.arbitratorKind = "SHARED";
    using ExtraConfig = memory::SharedArbitrator::ExtraConfig;
    options.extraArbitratorConfigs = {
        {std::string(ExtraConfig::kMemoryPoolInitialCapacity), "0B"},
        {std::string(ExtraConfig::kMemoryPoolReservedCapacity), "0B"}};
    std::atomic<int> arbitrations{0};
    options.arbitrationStateCheckCb = [&](memory::MemoryPool&) {
      ++arbitrations;
    };
    memory::MemoryManager manager(options);
    // Drain all driver closures before destroying this test-local manager.
    // A zero used-byte count alone does not mean Task/QueryCtx are destroyed.
    folly::CPUThreadPoolExecutor queryExecutor(4);
    auto root = manager.addRootPool(
        "cte-query", capacity, memory::MemoryReclaimer::create());
    auto ctx =
        core::QueryCtx::Builder().executor(&queryExecutor).pool(root).build();
    ASSERT_EQ(root->capacity(), 0);
    if (capacity == (8LL << 20)) {
      try {
        AssertQueryBuilder(plan).queryCtx(ctx).maxDrivers(1).assertResults(
            expected);
        FAIL() << "Expected query memory limit failure";
      } catch (const VeloxException& error) {
        EXPECT_EQ(error.errorCode(), error_code::kMemCapExceeded);
      }
    } else {
      AssertQueryBuilder(plan).queryCtx(ctx).maxDrivers(1).assertResults(
          expected);
      EXPECT_GT(root->peakBytes(), 64LL << 20);
    }
    EXPECT_GT(arbitrations.load(), 0);
    EXPECT_LE(root->capacity(), capacity);
    ctx.reset();
    queryExecutor.join();
    EXPECT_EQ(root->usedBytes(), 0);
  }
}

TEST_F(CteTest, explicitApiAndValidation) {
  auto ids = std::make_shared<core::PlanNodeIdGenerator>();
  auto batch = makeRowVector({makeFlatVector<int64_t>({1, 2, 3})});
  auto left = PlanBuilder(ids).values({batch}).planNode();
  auto right = PlanBuilder(ids).values({batch}).planNode();
  auto tree = PlanBuilder(ids).localPartition({}, {left, right}).planNode();
  EXPECT_EQ(finalizeCtes(tree), tree);
  EXPECT_EQ(
      finalizeCtes(tree, {0}), tree); // Irrelevant CTE options are ignored.
  auto dag = PlanBuilder(ids).localPartition({}, {left, left}).planNode();
  VELOX_ASSERT_THROW(finalizeCtes(dag), "Duplicate plan or CTE ID");
  Cte cte(ids->next(), left);
  auto scan = cte.scan(ids->next());
  AssertQueryBuilder(finalizeCtes(scan)).assertResults(batch);
  auto duplicate = PlanBuilder(ids).localPartition({}, {scan, scan}).planNode();
  VELOX_ASSERT_THROW(finalizeCtes(duplicate), "Create a separate scan");
  VELOX_ASSERT_THROW(Cte("", left), "CTE ID must not be empty");
  VELOX_ASSERT_THROW(Cte(ids->next(), nullptr), "CTE has no producer");
  Cte conflicting(left->id(), left);
  VELOX_ASSERT_THROW(
      finalizeCtes(conflicting.scan(ids->next())), "Duplicate plan or CTE ID");
  VELOX_ASSERT_THROW(
      finalizeCtes(finalizeCtes(scan)), "Finalize CTEs only once");
  VELOX_ASSERT_THROW(
      finalizeCtes(scan, {0}), "maxRetainedBytes must be positive");

  AssertQueryBuilder(finalizeCtes(scan))
      .config(core::QueryConfig::kSpillEnabled, true)
      .config(core::QueryConfig::kJoinSpillEnabled, true)
      .assertResults(batch);
}
} // namespace
} // namespace facebook::velox::exec::test
