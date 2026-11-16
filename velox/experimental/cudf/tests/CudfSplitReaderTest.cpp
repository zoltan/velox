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

#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReader.h"
#include "velox/experimental/cudf/exec/CudfMemoryArbitration.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/caching/FileHandle.h"
#include "velox/common/config/Config.h"
#include "velox/common/memory/MemoryArbitrator.h"

#include <cudf/ast/expressions.hpp>
#include <cudf/types.hpp>

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {
namespace {

class MetadataOnlySplitReader final : public CudfSplitReader {
 public:
  using CudfSplitReader::CudfSplitReader;

  cudf::ast::expression const* logicalFilter() const {
    return subfieldFilterAst();
  }

  cudf::ast::expression const* splitFilter() const {
    return pushdownFilter();
  }

  bool hasSplitFilter() const {
    return hasSplitSpecificPushdownFilter();
  }

 protected:
  void prepareSplitInternal(
      dwio::common::RuntimeStats& /*runtimeStats*/) override {}
};

class CudfSplitReaderTest : public ::facebook::velox::cudf_velox::exec::test::
                                CudfHiveConnectorTestBase {};

TEST_F(CudfSplitReaderTest, accountsFilterScalarsDuringSourceConstruction) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  FileHandleFactory files(
      std::make_unique<FileHandleCache>(1000),
      std::make_unique<FileHandleGenerator>());
  ColumnHandleMap columns{{"c0", makeColumnHandle("c0", BIGINT())}};
  for (const bool tracked : {false, true}) {
    SCOPED_TRACE(tracked);
    auto backend = createCudfCustomMemoryResource(32L << 20);
    auto root = memory::memoryManager()->addCustomRootPool(
        "filter-scalars-root", backend);
    auto gpuPool = root->addLeafChild("filter-scalars");
    auto builder = ::facebook::velox::connector::ConnectorQueryCtx::Builder();
    builder.operatorPool(pool_.get())
        .connectorPool(pool_.get())
        .sessionProperties(properties.get());
    if (tracked) {
      builder.customPools({{"gpu", gpuPool.get()}});
    }
    auto context = builder.build();
    common::SubfieldFilters filters;
    filters.emplace(
        common::Subfield("c0"),
        std::make_unique<common::BigintRange>(1, 3, false));
    auto table = makeTableHandle("filtered", rowType, std::move(filters));
    const auto baseline = cudfAllocatedBytes();
    auto source = std::make_unique<CudfHiveDataSource>(
        rowType,
        table,
        columns,
        &files,
        ioExecutor_.get(),
        context.get(),
        std::make_shared<CudfHiveConfig>(properties));
    EXPECT_GT(cudfAllocatedBytes(), baseline);
    if (tracked) {
      EXPECT_GT(gpuPool->usedBytes(), 0);
      EXPECT_NE(gpuPool->reclaimer(), nullptr);
      EXPECT_FALSE(gpuPool->reclaimableBytes().has_value());
      EXPECT_GT(backend->arbitrator()->stats().numRequests, 0);
    } else {
      EXPECT_EQ(gpuPool->usedBytes(), 0);
      EXPECT_EQ(gpuPool->reclaimer(), nullptr);
    }
    source.reset();
    EXPECT_EQ(gpuPool->usedBytes(), 0);
    EXPECT_EQ(cudfAllocatedBytes(), baseline);
  }
}

TEST_F(CudfSplitReaderTest, adoptedReaderOwnsResourcesWithoutQueryContext) {
  const auto rowType = ROW({"c0"}, {BIGINT()});
  constexpr vector_size_t kRows = 4096;
  constexpr int kGroups = 3;
  auto file = common::testutil::TempFilePath::create();
  std::vector<RowVectorPtr> inputs;
  for (int group = 0; group < kGroups; ++group) {
    inputs.push_back(makeRowVector(
        {"c0"}, {makeFlatVector<int64_t>(kRows, [group](vector_size_t i) {
          return group * kRows + i;
        })}));
  }
  writeToFile(file->getPath(), inputs);

  for (const bool buffered : {false, true}) {
    SCOPED_TRACE(buffered ? "buffered" : "direct");
    auto properties = std::make_shared<config::ConfigBase>(
        std::unordered_map<std::string, std::string>{
            {CudfHiveConfig::kUseBufferedInput, buffered ? "true" : "false"},
            {CudfHiveConfig::kMaxChunkReadLimit, "8192"},
            // Force later passes after the prepared source is destroyed.
            {CudfHiveConfig::kMaxPassReadLimit, "1"}});
    auto gpuRoot = memory::memoryManager()->addRootPool("adopted-reader");
    auto preparedPool = gpuRoot->addLeafChild("prepared-scan");
    auto activePool = gpuRoot->addLeafChild("active-scan");
    std::weak_ptr<memory::MemoryPool> weakPool = preparedPool;
    std::weak_ptr<memory::MemoryPool> weakActivePool = activePool;
    auto makeContext = [&](memory::MemoryPool* gpuPool) {
      return ::facebook::velox::connector::ConnectorQueryCtx::Builder()
          .operatorPool(pool_.get())
          .connectorPool(pool_.get())
          .sessionProperties(properties.get())
          .queryId("query.reader-handoff")
          .taskId("task.reader-handoff")
          .planNodeId("scan")
          .customPools({{"gpu", gpuPool}})
          .build();
    };
    auto preparedContext = makeContext(preparedPool.get());
    auto activeContext = makeContext(activePool.get());
    FileHandleFactory files(
        std::make_unique<FileHandleCache>(1000),
        std::make_unique<FileHandleGenerator>());
    ColumnHandleMap columns{{"c0", makeColumnHandle("c0", BIGINT())}};
    auto tableHandle = makeTableHandle("parquet_table", rowType);
    auto config = std::make_shared<CudfHiveConfig>(properties);
    auto makeSource = [&](const auto& context) {
      return std::make_unique<CudfHiveDataSource>(
          rowType,
          tableHandle,
          columns,
          &files,
          ioExecutor_.get(),
          context.get(),
          config);
    };
    auto prepared = makeSource(preparedContext);
    prepared->addSplit(makeCudfHiveConnectorSplit(file->getPath()));
    preparedContext.reset();
    auto active = makeSource(activeContext);
    active->setFromDataSource(std::move(prepared));
    RowVectorPtr retained;
    vector_size_t rows = 0;
    int chunks = 0;
    for (;;) {
      ContinueFuture future;
      auto result = active->next(kRows, future);
      ASSERT_TRUE(result.has_value());
      if (!*result)
        break;
      auto gpu = std::dynamic_pointer_cast<CudfVector>(*result);
      ASSERT_NE(gpu, nullptr);
      gpu->stream().sync();
      auto column = gpu->getTableView().column(0);
      std::vector<int64_t> actual(column.size());
      ASSERT_EQ(
          cudaMemcpy(
              actual.data(),
              column.data<int64_t>(),
              actual.size() * sizeof(int64_t),
              cudaMemcpyDeviceToHost),
          cudaSuccess);
      for (size_t i = 0; i < actual.size(); ++i)
        EXPECT_EQ(actual[i], rows + i);
      rows += gpu->size();
      ++chunks;
      if (!retained)
        retained = *result;
    }
    EXPECT_EQ(rows, kGroups * kRows);
    EXPECT_GT(chunks, 1);
    EXPECT_GT(gpuRoot->peakBytes(), 0);
    EXPECT_EQ(activePool->usedBytes(), 0);
    active.reset();
    activeContext.reset();
    preparedPool.reset();
    activePool.reset();
    gpuRoot.reset();
    ASSERT_FALSE(weakPool.expired());
    EXPECT_GT(weakPool.lock()->usedBytes(), 0);
    retained.reset();
    EXPECT_TRUE(weakPool.expired());
    EXPECT_TRUE(weakActivePool.expired());
  }
}

TEST_F(CudfSplitReaderTest, setsPushdownFilterAfterReadingMetadata) {
  auto rowType = ROW({"c0"}, {BIGINT()});
  auto dataFile = common::testutil::TempFilePath::create();
  writeToFile(
      dataFile->getPath(),
      makeNamedRowVector("c0", makeFlatVector<int64_t>({1, 2, 3})));

  auto properties = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>{});
  auto connectorQueryCtx =
      ::facebook::velox::connector::ConnectorQueryCtx::Builder()
          .operatorPool(pool_.get())
          .connectorPool(pool_.get())
          .sessionProperties(properties.get())
          .queryId("query.CudfSplitReaderTest")
          .taskId("task.CudfSplitReaderTest")
          .planNodeId("plan.CudfSplitReaderTest")
          .build();
  FileHandleFactory fileHandleFactory(
      std::make_unique<FileHandleCache>(1000),
      std::make_unique<FileHandleGenerator>());
  auto split =
      CudfHiveConnectorSplitBuilder(dataFile->getPath())
          .connectorId(
              ::facebook::velox::cudf_velox::exec::test::kCudfHiveConnectorId)
          .build();

  cudf::ast::column_reference logicalFilter{0};
  cudf::ast::column_reference splitFilter{0};
  MetadataOnlySplitReader reader(
      std::move(split),
      ::facebook::velox::cudf_velox::exec::test::CudfHiveConnectorTestBase::
          makeTableHandle("parquet_table", rowType),
      rowType,
      {"c0"},
      &fileHandleFactory,
      ioExecutor_.get(),
      connectorQueryCtx.get(),
      std::make_shared<CudfHiveConfig>(properties),
      std::make_shared<io::IoStatistics>(),
      std::make_shared<IoStats>(),
      &logicalFilter);

  EXPECT_EQ(reader.logicalFilter(), &logicalFilter);
  EXPECT_EQ(reader.splitFilter(), &logicalFilter);
  EXPECT_FALSE(reader.hasSplitFilter());

  const auto& fileMetadatas = reader.fileMetaDatas();
  ASSERT_FALSE(fileMetadatas.empty());
  EXPECT_GT(fileMetadatas.front().schema.size(), 1);
  EXPECT_EQ(&reader.fileMetaDatas(), &fileMetadatas);

  reader.setPushdownFilter(&splitFilter);
  dwio::common::RuntimeStats runtimeStats;
  // Footer inspection only allocates host metadata. Attach GPU resources before
  // preparation, and reject replacing them even before decoding starts.
  reader.setMemoryResources(*mr_, *output_mr_);
  EXPECT_THROW(reader.setMemoryResources(*mr_, *output_mr_), VeloxRuntimeError);
  reader.prepareSplit(runtimeStats);
  EXPECT_THROW(reader.setMemoryResources(*mr_, *output_mr_), VeloxRuntimeError);
  EXPECT_EQ(reader.logicalFilter(), &logicalFilter);
  EXPECT_EQ(reader.splitFilter(), &splitFilter);
  EXPECT_TRUE(reader.hasSplitFilter());
  EXPECT_EQ(runtimeStats.processedSplits, 1);
}

// The reader casts file timestamps to the session's
// parquet.reader.timestamp_type when the session sets it, and to the
// connector's parquet.reader.timestamp-type otherwise.
TEST_F(CudfSplitReaderTest, timestampTypeFollowsSession) {
  auto rowType = ROW({"c0"}, {TIMESTAMP()});
  auto dataFile = common::testutil::TempFilePath::create();
  writeToFile(
      dataFile->getPath(),
      makeNamedRowVector(
          "c0", makeFlatVector<Timestamp>({Timestamp(1, 0), Timestamp(2, 0)})));
  // Both settings take the enumerator's integer value.
  const auto seconds =
      std::to_string(static_cast<int32_t>(cudf::type_id::TIMESTAMP_SECONDS));
  const auto microseconds = std::to_string(
      static_cast<int32_t>(cudf::type_id::TIMESTAMP_MICROSECONDS));

  // Reads the file with the connector config 'connectorConfig' under
  // 'session' and returns the cuDF type of its column.
  const auto readTimestampType =
      [&](const std::unordered_map<std::string, std::string>& connectorConfig,
          const std::unordered_map<std::string, std::string>& session) {
        auto sessionProperties = std::make_shared<config::ConfigBase>(
            std::unordered_map<std::string, std::string>(session));
        auto connectorQueryCtx =
            ::facebook::velox::connector::ConnectorQueryCtx::Builder()
                .operatorPool(pool_.get())
                .connectorPool(pool_.get())
                .sessionProperties(sessionProperties.get())
                .queryId("query.CudfSplitReaderTest")
                .taskId("task.CudfSplitReaderTest")
                .planNodeId("plan.CudfSplitReaderTest")
                .build();
        FileHandleFactory fileHandleFactory(
            std::make_unique<FileHandleCache>(1000),
            std::make_unique<FileHandleGenerator>());
        auto split = CudfHiveConnectorSplitBuilder(dataFile->getPath())
                         .connectorId(
                             ::facebook::velox::cudf_velox::exec::test::
                                 kCudfHiveConnectorId)
                         .build();
        CudfSplitReader reader(
            std::move(split),
            ::facebook::velox::cudf_velox::exec::test::
                CudfHiveConnectorTestBase::makeTableHandle(
                    "parquet_table", rowType),
            rowType,
            {"c0"},
            &fileHandleFactory,
            ioExecutor_.get(),
            connectorQueryCtx.get(),
            std::make_shared<CudfHiveConfig>(
                std::make_shared<config::ConfigBase>(
                    std::unordered_map<std::string, std::string>(
                        connectorConfig))),
            std::make_shared<io::IoStatistics>(),
            std::make_shared<IoStats>(),
            nullptr);
        dwio::common::RuntimeStats runtimeStats;
        reader.prepareSplit(runtimeStats);
        auto chunk = reader.next(std::numeric_limits<uint64_t>::max());
        reader.stream().sync();
        if (!chunk.has_value()) {
          ADD_FAILURE() << "The reader returned no chunk";
          return cudf::type_id::EMPTY;
        }
        return chunk->table->view().column(0).type().id();
      };

  EXPECT_EQ(readTimestampType({}, {}), cudf::type_id::TIMESTAMP_MILLISECONDS);
  EXPECT_EQ(
      readTimestampType({{CudfHiveConfig::kTimestampType, seconds}}, {}),
      cudf::type_id::TIMESTAMP_SECONDS);
  EXPECT_EQ(
      readTimestampType(
          {{CudfHiveConfig::kTimestampType, seconds}},
          {{CudfHiveConfig::kTimestampTypeSession, microseconds}}),
      cudf::type_id::TIMESTAMP_MICROSECONDS);
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
