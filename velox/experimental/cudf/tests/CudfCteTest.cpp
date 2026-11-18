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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfAssignUniqueId.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/CudfGroupId.h"
#include "velox/experimental/cudf/exec/CudfMarkDistinct.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/SharedResultBuffer.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/Udf.h"

#include <cudf/contiguous_split.hpp>
#include <cudf/null_mask.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <cuda/stream>

#include <folly/ScopeGuard.h>

#include <array>
#include <future>

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::cudf_velox;

namespace {
template <typename T>
struct CpuOnlyCteIdentity {
  void call(int64_t& output, const int64_t& input) {
    output = input;
  }
};
} // namespace

class CudfCteTest : public OperatorTestBase {
 protected:
  void SetUp() override {
    OperatorTestBase::SetUp();
    savedResource_ = CudfConfig::getInstance().memoryResource;
    if (const auto* resource = std::getenv("VELOX_CTE_MEMORY_RESOURCE")) {
      CudfConfig::getInstance().memoryResource = resource;
    }
    savedFallback_ = CudfConfig::getInstance().allowCpuFallback;
    CudfConfig::getInstance().allowCpuFallback = false;
    registerCudf();
    Type::registerSerDe();
    core::ITypedExpr::registerSerDe();
    core::PlanNode::registerSerDe();
    registerCteSerDe();
  }
  void TearDown() override {
    unregisterCudf();
    CudfConfig::getInstance().allowCpuFallback = savedFallback_;
    CudfConfig::getInstance().memoryResource = savedResource_;
    OperatorTestBase::TearDown();
  }
  bool savedFallback_;
  std::string savedResource_;
};

TEST_F(CudfCteTest, sharedProjectionDoesNotCopyUnusedColumns) {
  constexpr vector_size_t kRows = 1 << 18;
  std::vector<VectorPtr> columns;
  for (int i = 0; i < 10; ++i) {
    columns.push_back(
        makeFlatVector<int64_t>(kRows, [](auto row) { return row; }));
  }
  auto host = makeRowVector(columns);
  auto plan = PlanBuilder().values({host}).project({"c0"}).planNode();
  auto project = std::dynamic_pointer_cast<const core::ProjectNode>(plan);
  auto task = Task::create(
      "shared-project",
      core::PlanFragment{plan->sources()[0]},
      0,
      core::QueryCtx::create(driverExecutor_.get()),
      Task::ExecutionMode::kParallel);
  auto driver =
      Driver::testingCreate(std::make_unique<DriverCtx>(task, 0, 0, 0, 0));
  cuda::stream stream{cuda::device_ref{0}};
  auto table = with_arrow::toCudfTable(host, pool(), stream, get_output_mr());
  auto original = std::make_shared<CudfVector>(
      pool(), host->type(), kRows, std::move(table), stream);
  CudfFilterProject op(0, driver->driverCtx(), nullptr, project);
  op.initialize();
  const auto inputBytes = cudfAllocatedBytes();
  statsMr_->push_counters();
  if (outputStatsMr_) {
    outputStatsMr_->push_counters();
  }
  const int iterations = std::getenv("VELOX_CTE_PROJECT_ITERATIONS")
      ? std::max(1, std::atoi(std::getenv("VELOX_CTE_PROJECT_ITERATIONS")))
      : 1;
  CudfVectorPtr result;
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    result.reset();
    op.addInput(original->shareForRead(pool()));
    result = std::dynamic_pointer_cast<CudfVector>(op.getOutput());
  }
  stream.sync();
  const auto elapsed = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - start)
                           .count();
  auto counters = statsMr_->pop_counters().first;
  auto allocated = counters.total;
  auto peak = counters.peak;
  if (outputStatsMr_) {
    auto outputCounters = outputStatsMr_->pop_counters().first;
    allocated += outputCounters.total;
    peak += outputCounters.peak;
  }
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(allocated, 0);
  EXPECT_EQ(
      result->getTableView().column(0).head<int64_t>(),
      original->getTableView().column(0).head<int64_t>());
  std::cout << "SHARED_PROJECT allocated_bytes=" << allocated
            << " peak_bytes=" << peak << " iterations=" << iterations
            << " elapsed_ms=" << elapsed << " input_live_bytes=" << inputBytes
            << " application_peak_bytes=" << inputBytes + peak << std::endl;
  auto expected = makeRowVector({columns[0]});
  auto actual = with_arrow::toVeloxColumn(
      result->getTableView(),
      pool(),
      expected->type(),
      stream,
      cudf::get_current_device_resource_ref());
  facebook::velox::test::assertEqualVectors(expected, actual);
  auto unchanged = with_arrow::toVeloxColumn(
      original->getTableView(),
      pool(),
      host->type(),
      stream,
      cudf::get_current_device_resource_ref());
  facebook::velox::test::assertEqualVectors(host, unchanged);
  op.close();
}

TEST_F(CudfCteTest, sharedFilterAndProjectionPreserveInput) {
  auto host = makeRowVector(
      {makeNullableFlatVector<int64_t>({0, 1, std::nullopt, 3}),
       makeFlatVector<int64_t>({10, 11, 12, 13})});
  createDuckDbTable({host});
  for (const auto& predicate :
       {"c1 >= 0", "c0 % 2 = 0", "c1 < 0", "c0 IS NULL"}) {
    SCOPED_TRACE(predicate);
    PlanBuilder query(pool());
    query.with("t", [&](auto& producer) {
      producer.values({host}).project({"c0", "c1"});
    });
    auto first = query.newPlan()
                     .fromCte("t")
                     .filter(predicate)
                     .project({"c0 AS x", "c0 AS y", "c1 + 1 AS z"})
                     .planNode();
    auto second = query.newPlan()
                      .fromCte("t")
                      .project({"c0 AS x", "c0 AS y", "c1 AS z"})
                      .planNode();
    auto plan = query.localPartition({}, {first, second}).build();
    AssertQueryBuilder(plan, duckDbQueryRunner_)
        .assertResults(
            fmt::format(
                "SELECT c0, c0, c1 + 1 FROM tmp WHERE {} UNION ALL SELECT c0, c0, c1 FROM tmp",
                predicate));
  }
}

// Device-only allocation/copy/publication/release cycle. The two consumer
// streams read the shared payload before the next iteration reuses pool space.
TEST_F(CudfCteTest, computedOnlySharedProjectionOwnsItsOutput) {
  auto host = makeRowVector({makeFlatVector<int64_t>({1, 2, 3, 4})});
  auto plan =
      PlanBuilder().values({host}).project({"c0 + 7 as added"}).planNode();
  auto task = Task::create(
      "computed-only",
      core::PlanFragment{plan->sources()[0]},
      0,
      core::QueryCtx::create(driverExecutor_.get()),
      Task::ExecutionMode::kParallel);
  auto driver =
      Driver::testingCreate(std::make_unique<DriverCtx>(task, 0, 0, 0, 0));
  CudfFilterProject op(
      0,
      driver->driverCtx(),
      nullptr,
      std::dynamic_pointer_cast<const core::ProjectNode>(plan));
  const auto baseline = cudfAllocatedBytes();
  op.initialize();
  cuda::stream stream{cuda::device_ref{0}};
  auto table = with_arrow::toCudfTable(host, pool(), stream, get_output_mr());
  auto original = std::make_shared<CudfVector>(
      pool(), host->type(), host->size(), std::move(table), stream);
  op.addInput(original->shareForRead(pool()));
  auto output = std::dynamic_pointer_cast<CudfVector>(op.getOutput());
  ASSERT_NE(output, nullptr);
  EXPECT_FALSE(output->isShared());
  original.reset();
  op.close();
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
  EXPECT_EQ(cudfAllocatedBytes(), baseline + output->estimateFlatSize());
  const auto* pointer = output->getTableView().column(0).head<int64_t>();
  auto owned = output->release(get_output_mr());
  EXPECT_EQ(owned->view().column(0).head<int64_t>(), pointer);
  auto expected = makeRowVector({makeFlatVector<int64_t>({8, 9, 10, 11})});
  auto actual = with_arrow::toVeloxColumn(
      owned->view(), pool(), expected->type(), stream, get_output_mr());
  facebook::velox::test::assertEqualVectors(expected, actual);
}

TEST_F(CudfCteTest, sharedColumnsSurviveInputAndStreamChange) {
  constexpr vector_size_t kRows = 512;
  auto keys = makeFlatVector<int64_t>(kRows, [](auto i) { return i % 16; });
  auto values = makeFlatVector<int64_t>(
      kRows, [](auto i) { return i; }, [](auto i) { return i % 7 == 0; });
  auto host = makeRowVector({keys, values});
  for (int kind = 0; kind < 4; ++kind) {
    SCOPED_TRACE(kind);
    auto builder = PlanBuilder().values({host});
    if (kind == 0) {
      builder.project({"c0", "c1", "c0 + 7 as added"});
    } else if (kind == 1) {
      builder.assignUniqueId();
    } else if (kind == 2) {
      builder.markDistinct("marker", {"c0"});
    } else {
      builder.groupId({"c0"}, {{"c0"}, {}}, {"c1"});
    }
    auto plan = builder.planNode();
    auto task = Task::create(
        "composite",
        core::PlanFragment{plan->sources()[0]},
        0,
        core::QueryCtx::create(driverExecutor_.get()),
        Task::ExecutionMode::kParallel);
    auto driver =
        Driver::testingCreate(std::make_unique<DriverCtx>(task, 0, 0, 0, 0));
    std::unique_ptr<Operator> op;
    if (kind == 0) {
      op = std::make_unique<CudfFilterProject>(
          0,
          driver->driverCtx(),
          nullptr,
          std::dynamic_pointer_cast<const core::ProjectNode>(plan));
    } else if (kind == 1) {
      op = std::make_unique<CudfAssignUniqueId>(
          0,
          driver->driverCtx(),
          std::dynamic_pointer_cast<const core::AssignUniqueIdNode>(plan),
          0,
          std::make_shared<std::atomic_int64_t>(0));
    } else if (kind == 2) {
      op = std::make_unique<CudfMarkDistinct>(
          0,
          driver->driverCtx(),
          std::dynamic_pointer_cast<const core::MarkDistinctNode>(plan));
    } else {
      op = std::make_unique<CudfGroupId>(
          0,
          driver->driverCtx(),
          std::dynamic_pointer_cast<const core::GroupIdNode>(plan));
    }
    op->initialize();
    cuda::stream producer{cuda::device_ref{0}};
    cuda::stream consumer{cuda::device_ref{0}};
    auto table =
        with_arrow::toCudfTable(host, pool(), producer, get_output_mr());
    auto original = std::make_shared<CudfVector>(
        pool(), host->type(), kRows, std::move(table), producer);
    op->addInput(original->shareForRead(pool()));
    auto output = std::dynamic_pointer_cast<CudfVector>(op->getOutput());
    ASSERT_NE(output, nullptr);
    EXPECT_TRUE(output->isShared());
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(
          output->getTableView().column(i).head<int64_t>(),
          original->getTableView().column(i).head<int64_t>());
      EXPECT_EQ(
          output->getTableView().column(i).null_mask(),
          original->getTableView().column(i).null_mask());
    }
    auto reader =
        std::dynamic_pointer_cast<CudfVector>(output->shareForRead(pool()));
    CudfVectorPtr second;
    if (kind == 3) {
      second = std::dynamic_pointer_cast<CudfVector>(op->getOutput());
      ASSERT_NE(second, nullptr);
      EXPECT_EQ(
          second->getTableView().column(1).head<int64_t>(),
          original->getTableView().column(1).head<int64_t>());
      EXPECT_EQ(op->getOutput(), nullptr);
    }
    original.reset();
    output.reset();
    op->close();
    EXPECT_TRUE(reader->rebindStream(consumer));
    std::vector<VectorPtr> expectedColumns{keys, values};
    if (kind == 2) {
      expectedColumns.push_back(
          makeFlatVector<bool>(kRows, [](auto i) { return i < 16; }));
    } else {
      expectedColumns.push_back(makeFlatVector<int64_t>(kRows, [kind](auto i) {
        return kind == 0 ? i % 16 + 7 : kind == 1 ? i : 0;
      }));
    }
    auto expected = makeRowVector(plan->outputType()->names(), expectedColumns);
    auto actual = with_arrow::toVeloxColumn(
        reader->getTableView(),
        pool(),
        expected->type(),
        consumer,
        get_output_mr());
    facebook::velox::test::assertEqualVectors(expected, actual);
    if (second) {
      second->rebindStream(consumer);
      auto emptyKeys = makeNullConstant(TypeKind::BIGINT, kRows);
      auto expectedSecond = makeRowVector(
          plan->outputType()->names(),
          {emptyKeys, values, makeFlatVector<int64_t>(kRows, [](auto) {
             return 1;
           })});
      auto actualSecond = with_arrow::toVeloxColumn(
          second->getTableView(),
          pool(),
          expectedSecond->type(),
          consumer,
          get_output_mr());
      facebook::velox::test::assertEqualVectors(expectedSecond, actualSecond);
    }
    auto exclusive = reader->release(get_output_mr());
    EXPECT_EQ(exclusive->num_rows(), kRows);
    consumer.sync();
  }
}

TEST_F(CudfCteTest, sharedStorageReuse) {
  constexpr vector_size_t kRows = 1 << 18;
  const int iterations = std::getenv("VELOX_CTE_REUSE_ITERATIONS")
      ? std::max(1, std::atoi(std::getenv("VELOX_CTE_REUSE_ITERATIONS")))
      : 8;
  cuda::stream producer{cuda::device_ref{0}};
  cuda::stream first{cuda::device_ref{0}};
  cuda::stream second{cuda::device_ref{0}};
  auto host =
      makeRowVector({makeFlatVector<int64_t>(kRows, [](auto i) { return i; })});
  auto source =
      with_arrow::toCudfTable(host, pool(), producer, get_output_mr());
  std::array<rmm::device_buffer, 2> copies{
      rmm::device_buffer(kRows * sizeof(int64_t), first, get_output_mr()),
      rmm::device_buffer(kRows * sizeof(int64_t), second, get_output_mr())};
  producer.sync();
  first.sync();
  second.sync();
  const auto baseline = cudfAllocatedBytes();
  statsMr_->push_counters();
  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    auto table = std::make_unique<cudf::table>(
        source->view(), producer, get_output_mr());
    auto original = std::make_shared<CudfVector>(
        pool(), host->type(), kRows, std::move(table), producer);
    for (int reader = 0; reader < 2; ++reader) {
      auto shared =
          std::dynamic_pointer_cast<CudfVector>(original->shareForRead(pool()));
      cuda::stream_ref stream = reader == 0 ? first : second;
      ASSERT_TRUE(shared->rebindStream(stream));
      EXPECT_EQ(
          shared->getTableView().column(0).head<int64_t>(),
          original->getTableView().column(0).head<int64_t>());
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          copies[reader].data(),
          shared->getTableView().column(0).head<int64_t>(),
          copies[reader].size(),
          cudaMemcpyDeviceToDevice,
          stream.get()));
    }
  }
  // Include final reclamation stream completion, as well as consumer copies.
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
  const auto seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
          .count();
  const auto counters = statsMr_->pop_counters().first;
  EXPECT_EQ(cudfAllocatedBytes(), baseline);
  std::cout << "CTE_REUSE resource=" << CudfConfig::getInstance().memoryResource
            << " iterations=" << iterations << " seconds=" << seconds
            << " peak_increment=" << counters.peak
            << " live=" << cudfAllocatedBytes() << std::endl;
  std::vector<int64_t> result(kRows);
  for (int reader = 0; reader < 2; ++reader) {
    auto& copy = copies[reader];
    cuda::stream_ref stream = reader == 0 ? first : second;
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        result.data(),
        copy.data(),
        copy.size(),
        cudaMemcpyDeviceToHost,
        stream.get()));
    stream.sync();
    for (int i = 0; i < kRows; ++i) {
      ASSERT_EQ(result[i], i);
    }
  }
}

TEST_F(CudfCteTest, readerReleaseDoesNotBlockProducerStream) {
  cuda::stream producer{cuda::device_ref{0}};
  cuda::stream consumer{cuda::device_ref{0}};
  auto host = makeRowVector({makeFlatVector<int64_t>({1, 2, 3, 4})});
  auto table = with_arrow::toCudfTable(
      host, pool(), producer, cudf::get_current_device_resource_ref());
  auto original = std::make_shared<CudfVector>(
      pool(), host->type(), host->size(), std::move(table), producer);
  auto reader =
      std::dynamic_pointer_cast<CudfVector>(original->shareForRead(pool()));
  ASSERT_TRUE(reader->rebindStream(consumer));
  std::promise<void> unblock;
  auto gate = unblock.get_future();
  CUDF_CUDA_TRY(cudaLaunchHostFunc(
      consumer.get(),
      [](void* value) { static_cast<std::future<void>*>(value)->wait(); },
      &gate));
  reader.reset();
  cudaEvent_t marker;
  CUDF_CUDA_TRY(cudaEventCreateWithFlags(&marker, cudaEventDisableTiming));
  CUDF_CUDA_TRY(cudaEventRecord(marker, producer.get()));
  auto completion = std::async(
      std::launch::async, [marker] { return cudaEventSynchronize(marker); });
  const auto status = completion.wait_for(std::chrono::seconds(2));
  unblock.set_value();
  EXPECT_EQ(completion.get(), cudaSuccess);
  EXPECT_EQ(status, std::future_status::ready);
  cudaEventDestroy(marker);
  consumer.sync();
}

TEST_F(CudfCteTest, rootPassThroughFollowsCpuInput) {
  registerFunction<CpuOnlyCteIdentity, int64_t, int64_t>({"cte_cpu_only"});
  auto data = makeRowVector({makeFlatVector<int64_t>({1, 2, 3, 4})});
  PlanBuilder query(pool());
  query.with("shared", [&](auto& producer) {
    producer.values({data}).project({"c0"});
  });
  auto plan =
      query.fromCte("shared").project({"cte_cpu_only(c0) AS c0"}).build();
  auto task = AssertQueryBuilder(plan)
                  .config(CudfConfig::kCudfAllowCpuFallback, true)
                  .assertResults(data);
  const auto stats = toPlanStats(task->taskStats());
  const auto& root = stats.at(plan->id());
  EXPECT_EQ(root.operatorStats.count("CudfFromVelox"), 0);
  EXPECT_EQ(root.operatorStats.count("CudfToVelox"), 0);
}

TEST_F(CudfCteTest, matchingDomainConversionIsZeroCopy) {
  auto host = makeRowVector({makeFlatVector<int64_t>({1, 2, 3, 4})});
  auto plan = PlanBuilder().values({host}).planNode();
  auto task = Task::create(
      "matching-cte",
      core::PlanFragment{plan},
      0,
      core::QueryCtx::create(driverExecutor_.get()),
      Task::ExecutionMode::kParallel);
  auto driver =
      Driver::testingCreate(std::make_unique<DriverCtx>(task, 0, 0, 0, 0));
  cuda::stream stream{cuda::device_ref{0}};
  auto table = with_arrow::toCudfTable(host, pool(), stream, get_output_mr());
  auto original = std::make_shared<CudfVector>(
      pool(), host->type(), host->size(), std::move(table), stream);
  for (const bool gpu : {false, true}) {
    RowVectorPtr input = gpu
        ? std::static_pointer_cast<RowVector>(original->shareForRead(pool()))
        : host;
    const auto bytes = cudfAllocatedBytes();
    std::unique_ptr<Operator> conversion;
    if (gpu) {
      auto op = std::make_unique<CudfFromVelox>(
          0, asRowType(host->type()), driver->driverCtx(), "matching");
      op->allowMatchingInput();
      conversion = std::move(op);
    } else {
      auto op = std::make_unique<CudfToVelox>(
          0, asRowType(host->type()), driver->driverCtx(), "matching");
      op->allowMatchingInput();
      conversion = std::move(op);
    }
    conversion->initialize();
    const int iterations = std::getenv("VELOX_CTE_CONVERSION_ITERATIONS")
        ? std::max(1, std::atoi(std::getenv("VELOX_CTE_CONVERSION_ITERATIONS")))
        : 2;
    bool same = true;
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
      conversion->addInput(input);
      same &= !conversion->needsInput();
      same &= conversion->getOutput().get() == input.get();
    }
    const auto seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
            .count();
    EXPECT_TRUE(same);
    std::cout << "CTE_CONVERSION gpu=" << gpu << " iterations=" << iterations
              << " seconds=" << seconds << std::endl;
    EXPECT_EQ(cudfAllocatedBytes(), bytes);
    conversion->noMoreInput();
    EXPECT_EQ(conversion->getOutput(), nullptr);
    EXPECT_TRUE(conversion->isFinished());
    conversion->close();
    EXPECT_EQ(
        conversion->stats().rlock()->runtimeStats.at("matchingInputRows").sum,
        iterations * input->size());
    conversion->close();
    EXPECT_EQ(
        conversion->stats().rlock()->runtimeStats.at("matchingInputRows").sum,
        iterations * input->size());
  }
}

TEST_F(CudfCteTest, matchingDomainPreservesBatching) {
  auto host = makeRowVector({makeFlatVector<int64_t>({1, 2, 3, 4})});
  auto plan = PlanBuilder().values({host}).planNode();
  auto task = Task::create(
      "batching-cte",
      core::PlanFragment{plan},
      0,
      core::QueryCtx::create(driverExecutor_.get()),
      Task::ExecutionMode::kParallel);
  auto driver =
      Driver::testingCreate(std::make_unique<DriverCtx>(task, 0, 0, 0, 0));
  cuda::stream stream{cuda::device_ref{0}};
  auto table = with_arrow::toCudfTable(host, pool(), stream, get_output_mr());
  auto device = std::make_shared<CudfVector>(
      pool(), host->type(), host->size(), std::move(table), stream);
  CudfFromVelox upload(
      0, asRowType(host->type()), driver->driverCtx(), "batching");
  upload.allowMatchingInput();
  upload.initialize();
  upload.addInput(host);
  EXPECT_EQ(upload.getOutput(), nullptr);
  // Matching input flushes accumulated host rows before forwarding itself.
  auto shared =
      std::static_pointer_cast<RowVector>(device->shareForRead(pool()));
  upload.addInput(shared);
  auto converted = upload.getOutput();
  ASSERT_NE(converted, nullptr);
  EXPECT_NE(converted.get(), shared.get());
  EXPECT_EQ(converted->size(), host->size());
  upload.noMoreInput();
  EXPECT_FALSE(upload.isFinished());
  EXPECT_EQ(upload.getOutput().get(), shared.get());
  EXPECT_EQ(upload.getOutput(), nullptr);
  EXPECT_TRUE(upload.isFinished());
  upload.close();

  CudfToVelox download(
      1, asRowType(host->type()), driver->driverCtx(), "batching");
  download.allowMatchingInput();
  download.initialize();
  download.addInput(shared);
  download.addInput(host);
  download.noMoreInput();
  auto output = download.getOutput();
  ASSERT_NE(output, nullptr);
  facebook::velox::test::assertEqualVectors(host, output);
  EXPECT_FALSE(download.isFinished());
  EXPECT_EQ(download.getOutput().get(), host.get());
  EXPECT_EQ(download.getOutput(), nullptr);
  EXPECT_TRUE(download.isFinished());
  download.close();
}

TEST_F(CudfCteTest, cpuProducerWithCpuReadersStaysOnHost) {
  registerFunction<CpuOnlyCteIdentity, int64_t, int64_t>({"cte_cpu_only"});
  auto data = makeRowVector({makeFlatVector<int64_t>({1, 2, 3, 4})});
  PlanBuilder query(pool());
  query.with("shared", [&](auto& producer) {
    producer.values({data}).project({"cte_cpu_only(c0) AS c0"});
  });
  auto plan =
      query.fromCte("shared").project({"cte_cpu_only(c0) AS c0"}).build();
  auto task = AssertQueryBuilder(plan)
                  .config(CudfConfig::kCudfAllowCpuFallback, true)
                  .assertResults(data);
  for (const auto& pipeline : task->taskStats().pipelineStats) {
    for (const auto& op : pipeline.operatorStats) {
      EXPECT_NE(op.operatorType, "CudfFromVelox");
    }
  }
}

TEST_F(CudfCteTest, zeroCopyBuffer) {
  cuda::stream stream{cuda::device_ref{0}};
  auto host = makeRowVector({makeFlatVector<int64_t>({1, 2, 3, 4})});
  auto table = with_arrow::toCudfTable(
      host, pool(), stream, cudf::get_current_device_resource_ref());
  auto input = std::make_shared<CudfVector>(
      pool(), host->type(), host->size(), std::move(table), stream);
  const auto* address = input->getTableView().column(0).head<int64_t>();
  const auto bytes = input->retainedSize();
  auto buffer = std::make_shared<exec::detail::SharedResultBuffer>(
      std::vector<core::PlanNodeId>{"first", "second"}, CteOptions{});
  buffer->setProducerCount(1);
  buffer->setReaderDriverCount(0, 1);
  buffer->setReaderDriverCount(1, 1);
  buffer->add(input);
  auto first = std::dynamic_pointer_cast<CudfVector>(buffer->next(0, pool()));
  ASSERT_NE(first, nullptr);
  EXPECT_NE(first.get(), input.get());
  EXPECT_EQ(first->getTableView().column(0).head<int64_t>(), address);
  EXPECT_EQ(first->retainedSize(), bytes);
  input.reset();
  auto second = std::dynamic_pointer_cast<CudfVector>(buffer->next(1, pool()));
  ASSERT_NE(second, nullptr);
  EXPECT_NE(first.get(), second.get());
  EXPECT_EQ(second->getTableView().column(0).head<int64_t>(), address);
  buffer->finish();
  buffer.reset();
  // An exclusive-ownership consumer must not steal another reader's storage.
  VELOX_ASSERT_THROW(first->release(), "requires a destination resource");
  auto owned = first->release(get_output_mr());
  first.reset();
  EXPECT_NE(owned->view().column(0).head<int64_t>(), address);
  EXPECT_EQ(second->getTableView().column(0).head<int64_t>(), address);
  auto result = with_arrow::toVeloxColumn(
      second->getTableView(),
      pool(),
      host->type(),
      stream,
      cudf::get_current_device_resource_ref());
  facebook::velox::test::assertEqualVectors(host, result);
}

TEST_F(CudfCteTest, sharedJoin) {
  auto data = makeRowVector(
      {makeFlatVector<int64_t>(1024, [](auto row) { return row; })});
  for (auto drivers : {1, 4}) {
    PlanBuilder query(pool());
    query.with("shared", [&](auto& producer) {
      producer.values({data}, true).project({"c0 + 1 as key"});
    });
    auto build =
        query.newPlan().fromCte("shared").project({"key as other"}).planNode();
    auto plan = query.fromCte("shared")
                    .hashJoin({"key"}, {"other"}, build, "", {"key"})
                    .localPartition({})
                    .singleAggregation({}, {"sum(key) as total"})
                    .build();
    AssertQueryBuilder(plan).maxDrivers(drivers).assertResults(makeRowVector(
        {makeFlatVector<int64_t>({524800LL * drivers * drivers})}));
  }
}

TEST_F(CudfCteTest, packedAndNestedStorage) {
  for (bool packed : {false, true}) {
    SCOPED_TRACE(packed);
    cuda::stream stream{cuda::device_ref{0}};
    rmm::mr::statistics_resource_adaptor resource{
        rmm::mr::cuda_async_memory_resource{}};
    auto memory = rmm::to_device_async_resource_ref_checked(&resource);
    auto host = makeRowVector(
        {makeNullableFlatVector<int64_t>({1, std::nullopt, 3, 4}),
         makeNullableFlatVector<StringView>(
             {"one", "two", std::nullopt, "four"}),
         makeArrayVector<int64_t>({{1, 2}, {}, {3}, {4, 5, 6}})});
    auto table = with_arrow::toCudfTable(host, pool(), stream, memory);
    CudfVectorPtr input;
    if (packed) {
      auto packedColumns = cudf::pack(table->view(), stream, memory);
      auto view = cudf::unpack(packedColumns);
      auto storage = std::make_unique<cudf::packed_table>(
          cudf::packed_table{view, std::move(packedColumns)});
      table.reset();
      input = std::make_shared<CudfVector>(
          pool(), host->type(), host->size(), std::move(storage), stream);
    } else {
      input = std::make_shared<CudfVector>(
          pool(), host->type(), host->size(), std::move(table), stream);
    }
    const auto allocationCount = resource.get_allocations_counter().total;
    const auto liveBytes = resource.get_bytes_counter().value;
    std::cout << "CTE_SHARING packed=" << packed
              << " payload_bytes=" << liveBytes
              << " allocations=" << allocationCount << std::endl;
    auto buffer = std::make_shared<exec::detail::SharedResultBuffer>(
        std::vector<core::PlanNodeId>{"first", "second"}, CteOptions{});
    buffer->setProducerCount(1);
    buffer->setReaderDriverCount(0, 1);
    buffer->setReaderDriverCount(1, 1);
    buffer->add(input);
    auto first = std::dynamic_pointer_cast<CudfVector>(buffer->next(0, pool()));
    ASSERT_NE(first, nullptr);
    buffer->closeReader(1);
    buffer->finish();
    buffer.reset();
    EXPECT_EQ(resource.get_allocations_counter().total, allocationCount);
    EXPECT_EQ(resource.get_bytes_counter().value, liveBytes);
    for (int column = 0; column < 3; ++column) {
      EXPECT_EQ(
          first->getTableView().column(column).head<void>(),
          input->getTableView().column(column).head<void>());
      EXPECT_EQ(
          first->getTableView().column(column).null_mask(),
          input->getTableView().column(column).null_mask());
    }
    input.reset();
    EXPECT_EQ(resource.get_bytes_counter().value, liveBytes);
    auto result = with_arrow::toVeloxColumn(
        first->getTableView(),
        pool(),
        host->type(),
        stream,
        cudf::get_current_device_resource_ref());
    facebook::velox::test::assertEqualVectors(host, result);
    first.reset();
    stream.sync();
    EXPECT_EQ(resource.get_bytes_counter().value, 0);
  }
}

TEST_F(CudfCteTest, asynchronousReaders) {
  constexpr int32_t rows = 1 << 20;
  constexpr int readers = 8;
  cuda::stream producer{cuda::device_ref{0}};
  rmm::mr::statistics_resource_adaptor resource{
      rmm::mr::cuda_async_memory_resource{}};
  auto memory = rmm::to_device_async_resource_ref_checked(&resource);
  rmm::device_buffer data(rows * sizeof(int64_t), producer, memory);
  ASSERT_EQ(
      cudaMemsetAsync(data.data(), 7, data.size(), producer.get()),
      cudaSuccess);
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
      std::make_unique<cudf::column>(
          cudf::data_type{cudf::type_id::INT64},
          rows,
          std::move(data),
          cudf::create_null_mask(
              0, cudf::mask_state::UNALLOCATED, producer, memory),
          0));
  auto input = std::make_shared<CudfVector>(
      pool(),
      ROW({"c0"}, {BIGINT()}),
      rows,
      std::make_unique<cudf::table>(std::move(columns)),
      producer);
  auto buffer = std::make_shared<exec::detail::SharedResultBuffer>(
      std::vector<core::PlanNodeId>{"0", "1", "2", "3", "4", "5", "6", "7"},
      CteOptions{});
  buffer->setProducerCount(1);
  for (int reader = 0; reader < readers; ++reader) {
    buffer->setReaderDriverCount(reader, 1);
  }
  buffer->add(input);
  input.reset();
  buffer->finish();
  std::vector<std::future<void>> workers;
  for (int reader = 0; reader < readers; ++reader) {
    workers.push_back(std::async(std::launch::async, [&, reader] {
      ensureCudaContextForThread();
      cuda::stream stream{cuda::device_ref{0}};
      auto output =
          std::dynamic_pointer_cast<CudfVector>(buffer->next(reader, pool()));
      ASSERT_NE(output, nullptr);
      ASSERT_TRUE(output->rebindStream(stream));
      rmm::device_buffer copied(rows * sizeof(int64_t), stream, memory);
      ASSERT_EQ(
          cudaMemcpyAsync(
              copied.data(),
              output->getTableView().column(0).head<int64_t>(),
              copied.size(),
              cudaMemcpyDeviceToDevice,
              stream.get()),
          cudaSuccess);
      // Drop the lease before the asynchronous read completes.
      output.reset();
      std::array<int64_t, 1> value;
      ASSERT_EQ(
          cudaMemcpyAsync(
              value.data(),
              copied.data(),
              sizeof(int64_t),
              cudaMemcpyDeviceToHost,
              stream.get()),
          cudaSuccess);
      stream.sync();
      EXPECT_EQ(value[0], 0x0707070707070707LL);
      buffer->closeReader(reader);
    }));
  }
  for (auto& worker : workers) {
    worker.get();
  }
  buffer->cancel();
  buffer.reset();
  producer.sync();
  EXPECT_EQ(resource.get_bytes_counter().value, 0);
}

TEST_F(CudfCteTest, retainedByteLimit) {
  cuda::stream stream{cuda::device_ref{0}};
  auto host = makeRowVector(
      {makeFlatVector<int64_t>(1024, [](auto row) { return row; })});
  auto table = with_arrow::toCudfTable(
      host, pool(), stream, cudf::get_current_device_resource_ref());
  auto input = std::make_shared<CudfVector>(
      pool(), host->type(), host->size(), std::move(table), stream);
  auto buffer = std::make_shared<exec::detail::SharedResultBuffer>(
      std::vector<core::PlanNodeId>{"first"}, CteOptions{4096});
  buffer->setProducerCount(1);
  buffer->setReaderDriverCount(0, 1);
  VELOX_ASSERT_THROW(buffer->add(input), "retained");
}

TEST_F(CudfCteTest, sharedRevenueAndSerializedPlan) {
  auto data = makeRowVector(
      {makeFlatVector<int64_t>(4096, [](auto row) { return row % 8; }),
       makeFlatVector<int64_t>(4096, [](auto row) { return row % 8 + 1; })});
  for (bool serialize : {false, true}) {
    PlanBuilder query(pool());
    query.with("revenue", [&](auto& producer) {
      producer.values({data}).singleAggregation({"c0"}, {"sum(c1) as revenue"});
    });
    auto maximum = query.newPlan()
                       .fromCte("revenue")
                       .localPartition({})
                       .singleAggregation({}, {"max(revenue) as maximum"})
                       .planNode();
    auto plan =
        query.fromCte("revenue")
            .hashJoin({"revenue"}, {"maximum"}, maximum, "", {"c0", "revenue"})
            .build();
    if (serialize) {
      plan =
          ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());
    }
    auto task =
        AssertQueryBuilder(plan).maxDrivers(4).assertResults(makeRowVector(
            {makeFlatVector<int64_t>({7}), makeFlatVector<int64_t>({4096})}));
    int64_t published = 0;
    int64_t consumed = 0;
    for (const auto& pipeline : task->taskStats().pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        if (op.operatorType == "SharedResultSink")
          published += op.inputPositions;
        if (op.operatorType == "SharedScan")
          consumed += op.outputPositions;
      }
    }
    int uploads = 0;
    int downloads = 0;
    for (const auto& pipeline : task->taskStats().pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        const auto matching = op.runtimeStats.find("matchingInputRows");
        const auto forwarded =
            matching == op.runtimeStats.end() ? 0 : matching->second.sum;
        // A reader's domain adapter forwards an already matching batch.
        // Count only edges that actually transfer input between host/device.
        if (op.inputPositions > forwarded) {
          uploads += op.operatorType == "CudfFromVelox";
          downloads += op.operatorType == "CudfToVelox";
        }
      }
    }
    EXPECT_EQ(uploads, 1);
    EXPECT_EQ(downloads, 1);
    EXPECT_EQ(published, 8);
    EXPECT_EQ(consumed, 16);
  }
}

TEST_F(CudfCteTest, cpuProducerAndDependentCte) {
  auto data = makeRowVector({makeFlatVector<int64_t>({1, 2, 3, 4})});
  PlanBuilder query(pool());
  query.with("base", [&](auto& producer) { producer.values({data}); });
  query.with("dependent", [&](auto& producer) {
    producer.fromCte("base").project({"c0 + 10 as other"});
  });
  auto build = query.newPlan().fromCte("dependent").planNode();
  auto plan = query.fromCte("base")
                  .project({"c0 + 10 as key"})
                  .hashJoin({"key"}, {"other"}, build, "", {"key"})
                  .build();
  AssertQueryBuilder(plan).maxDrivers(4).assertResults(
      makeRowVector({makeFlatVector<int64_t>({11, 12, 13, 14})}));
}

TEST_F(CudfCteTest, earlyLimit) {
  auto data = makeRowVector(
      {makeFlatVector<int64_t>(1024, [](auto row) { return row; })});
  PlanBuilder query(pool());
  query.with("shared", [&](auto& producer) {
    producer.values({data, data, data}).project({"c0 + 1 as key"});
  });
  auto left = query.newPlan().fromCte("shared").limit(0, 7, false).planNode();
  auto right = query.newPlan().fromCte("shared").limit(0, 11, false).planNode();
  auto plan = query.localPartition({}, {left, right})
                  .singleAggregation({}, {"count(1) as n"})
                  .build();
  AssertQueryBuilder(plan).maxDrivers(4).assertResults(
      makeRowVector({makeFlatVector<int64_t>({18})}));
}

TEST_F(CudfCteTest, unsupportedProducerStaysOnCpu) {
  auto data = makeRowVector(
      {makeFlatVector<int64_t>({1, 2, 3, 4}),
       makeMapVector<int64_t, int64_t>({{{1, 2}}, {}, {}, {{3, 4}}})});
  PlanBuilder query(pool());
  query.with("shared", [&](auto& producer) {
    producer.values({data}).project({"c0"});
  });
  auto plan =
      query.fromCte("shared").singleAggregation({}, {"sum(c0)"}).build();
  AssertQueryBuilder(plan)
      .config(CudfConfig::kCudfAllowCpuFallback, true)
      .assertResults(makeRowVector({makeFlatVector<int64_t>({10})}));
}

TEST_F(CudfCteTest, cpuProducerSchemaDiffersFromResult) {
  auto data = makeRowVector(
      {makeFlatVector<int64_t>({1, 2}), makeFlatVector<int64_t>({10, 20})});
  PlanBuilder query(pool());
  query.with("shared", [&](auto& producer) { producer.values({data}); });
  auto plan = query.fromCte("shared")
                  .project({"c0 + c1 as value"})
                  .singleAggregation({}, {"sum(value)"})
                  .build();
  AssertQueryBuilder(plan).assertResults(
      makeRowVector({makeFlatVector<int64_t>({33})}));
}

TEST_F(CudfCteTest, supportedProducerWithUnsupportedResult) {
  auto data = makeRowVector({makeFlatVector<int64_t>({1, 2})});
  auto expected =
      makeRowVector({makeMapVector<int64_t, int64_t>({{{2, 2}}, {{3, 3}}})});
  for (bool serialize : {false, true}) {
    SCOPED_TRACE(serialize);
    PlanBuilder query(pool());
    query.with("shared", [&](auto& producer) { producer.values({data}); });
    auto plan = query.fromCte("shared")
                    .project({"c0 + 1 as key"})
                    .localPartition({})
                    .project({"map(array[key], array[key]) as result"})
                    .build();
    if (serialize) {
      plan =
          ISerializable::deserialize<core::PlanNode>(plan->serialize(), pool());
    }
    AssertQueryBuilder(plan)
        .config(CudfConfig::kCudfAllowCpuFallback, true)
        .maxDrivers(4)
        .assertResults(expected);
  }
}

TEST_F(CudfCteTest, ownershipCopyUsesOutputResource) {
  cuda::stream stream{cuda::device_ref{0}};
  rmm::mr::statistics_resource_adaptor producerResource{
      rmm::mr::cuda_async_memory_resource{}};
  rmm::mr::statistics_resource_adaptor temporaryResource{
      rmm::mr::cuda_async_memory_resource{}};
  rmm::mr::statistics_resource_adaptor outputResource{
      rmm::mr::cuda_async_memory_resource{}};
  auto producerMr =
      rmm::to_device_async_resource_ref_checked(&producerResource);
  auto outputMr = rmm::to_device_async_resource_ref_checked(&outputResource);
  auto previousResource = cudf::set_current_device_resource(temporaryResource);
  SCOPE_EXIT {
    cudf::set_current_device_resource(std::move(previousResource));
  };

  auto host = makeRowVector({makeFlatVector<int64_t>({1, 2, 3, 4})});
  auto table = with_arrow::toCudfTable(host, pool(), stream, producerMr);
  auto input = std::make_shared<CudfVector>(
      pool(), host->type(), host->size(), std::move(table), stream);
  const auto* address = input->getTableView().column(0).head<int64_t>();
  const auto allocations = producerResource.get_allocations_counter().total;
  const auto bytes = producerResource.get_bytes_counter().value;
  auto first =
      std::dynamic_pointer_cast<CudfVector>(input->shareForRead(pool()));
  auto second =
      std::dynamic_pointer_cast<CudfVector>(input->shareForRead(pool()));
  input.reset();
  EXPECT_EQ(producerResource.get_allocations_counter().total, allocations);
  EXPECT_EQ(producerResource.get_bytes_counter().value, bytes);
  EXPECT_EQ(temporaryResource.get_allocations_counter().total, 0);
  EXPECT_EQ(outputResource.get_allocations_counter().total, 0);

  auto owned = first->release(outputMr);
  first.reset();
  EXPECT_EQ(temporaryResource.get_allocations_counter().total, 0);
  EXPECT_GT(outputResource.get_bytes_counter().value, 0);
  EXPECT_EQ(producerResource.get_bytes_counter().value, bytes);
  EXPECT_NE(owned->view().column(0).head<int64_t>(), address);
  EXPECT_EQ(second->getTableView().column(0).head<int64_t>(), address);
  auto result = with_arrow::toVeloxColumn(
      owned->view(), pool(), host->type(), stream, outputMr);
  facebook::velox::test::assertEqualVectors(host, result);

  owned.reset();
  second.reset();
  stream.sync();
  EXPECT_EQ(producerResource.get_bytes_counter().value, 0);
  EXPECT_EQ(temporaryResource.get_bytes_counter().value, 0);
  EXPECT_EQ(outputResource.get_bytes_counter().value, 0);
}

TEST_F(CudfCteTest, cancellationRetainsClaimedStorage) {
  cuda::stream producer{cuda::device_ref{0}};
  cuda::stream consumer{cuda::device_ref{0}};
  rmm::mr::statistics_resource_adaptor resource{
      rmm::mr::cuda_async_memory_resource{}};
  auto host = makeRowVector({makeFlatVector<int64_t>({1, 2, 3, 4})});
  auto table = with_arrow::toCudfTable(
      host,
      pool(),
      producer,
      rmm::to_device_async_resource_ref_checked(&resource));
  auto input = std::make_shared<CudfVector>(
      pool(), host->type(), host->size(), std::move(table), producer);
  auto buffer = std::make_shared<exec::detail::SharedResultBuffer>(
      std::vector<core::PlanNodeId>{"first", "lagging"}, CteOptions{});
  buffer->setProducerCount(1);
  buffer->setReaderDriverCount(0, 1);
  buffer->setReaderDriverCount(1, 1);
  buffer->add(input);
  auto claimed = std::dynamic_pointer_cast<CudfVector>(buffer->next(0, pool()));
  ASSERT_NE(claimed, nullptr);
  buffer->cancel();
  buffer.reset();
  input.reset();
  // Arrow conversion may retain an all-valid null mask as well as data.
  EXPECT_EQ(resource.get_bytes_counter().value, claimed->retainedSize());
  ASSERT_TRUE(claimed->rebindStream(consumer));
  auto result = with_arrow::toVeloxColumn(
      claimed->getTableView(),
      pool(),
      host->type(),
      consumer,
      cudf::get_current_device_resource_ref());
  facebook::velox::test::assertEqualVectors(host, result);
  claimed.reset();
  producer.sync();
  EXPECT_EQ(resource.get_bytes_counter().value, 0);
}
