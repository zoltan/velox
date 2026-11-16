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
#include "velox/experimental/cudf/benchmarks/CudfTpcdsBenchmark.h"
#include "velox/experimental/cudf/benchmarks/CudfTpchBenchmark.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/CudfMemoryArbitration.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/utils/CudfTpcdsQueryBuilder.h"

#include "velox/common/caching/AsyncDataCache.h"
#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/core/QueryCtx.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

#include <folly/ScopeGuard.h>
#include <folly/json.h>

#include <fstream>

DECLARE_string(data_path);
DECLARE_string(data_format);
DECLARE_int64(max_coalesced_bytes);
DECLARE_string(max_coalesced_distance_bytes);
DECLARE_int32(parquet_prefetch_rowgroups);
DECLARE_int32(num_drivers);
DECLARE_int32(num_repeats);

DEFINE_int32(suite_passes, 0, "Number of sequential passes over saved plans.");
DEFINE_string(
    suite_queries,
    "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22",
    "Comma-separated query IDs in execution order.");
DEFINE_string(
    suite_output,
    "",
    "JSONL output for complete query results and metrics.");
DEFINE_uint64(
    gpu_capacity_bytes,
    0,
    "Shared GPU accounting capacity; zero disables accounting.");
DEFINE_bool(
    suite_memory,
    false,
    "Measure per-query live allocation peaks; use a separate run for timing.");

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::dwio::common;

DEFINE_uint64(
    cudf_chunk_read_limit,
    0,
    "Output table chunk read limit for cudf::parquet_chunked_reader.");

DEFINE_uint64(
    cudf_pass_read_limit,
    0,
    "Pass read limit for cudf::parquet_chunked_reader.");

DEFINE_int32(
    cudf_gpu_batch_size_rows,
    100000,
    "Preferred output batch size in rows for cudf operators.");

DEFINE_bool(velox_cudf_table_scan, true, "Enable cuDF table scan");

DEFINE_string(
    cudf_properties,
    "",
    "Path to a properties file for CudfConfig. Each line should be key=value "
    "(e.g. cudf.memory_resource=async). See CudfConfig for available keys.");

void CudfTpcdsBenchmark::initQueryBuilder() {
  auto cudfBuilder =
      std::make_unique<cudf_velox::exec::test::CudfTpcdsQueryBuilder>(
          toFileFormat(FLAGS_data_format), ioExecutor_.get());
  cudfBuilder->enableCudf();
  cudfBuilder->initialize(FLAGS_data_path);
  queryBuilder_ = std::move(cudfBuilder);
}

void CudfTpcdsBenchmark::initialize() {
  // Must be set before TpcdsBenchmark::initialize(), which calls
  // initQueryBuilder() -> registerCudf(), which bakes the prefix into the
  // step-aware aggregation registry.
  auto& config = cudf_velox::CudfConfig::getInstance();
  config.functionNamePrefix = kPrestoFunctionNamespacePrefix;

  if (!FLAGS_cudf_properties.empty()) {
    config.initialize(cudf_velox::loadPropertiesFile(FLAGS_cudf_properties));
  }

  TpcdsBenchmark::initialize();
  if (FLAGS_gpu_capacity_bytes > 0) {
    cudf_velox::registerCudfMemoryResource(FLAGS_gpu_capacity_bytes);
    queryExecutor_ = std::make_unique<folly::CPUThreadPoolExecutor>(
        std::max(1, FLAGS_num_drivers));
  }

  const std::string prestoConnectorId{kPrestoHiveConnectorId};
  if (FLAGS_velox_cudf_table_scan) {
    // The base class registered a HiveConnector. Unregister it so the
    // CudfTpcdsQueryBuilder can register CudfHiveConnector under the plan's
    // connector ID instead. The query builder handles this when getQueryPlan()
    // is called.
    if (connector::ConnectorRegistry::tryGet(prestoConnectorId) != nullptr) {
      connector::ConnectorRegistry::global().erase(prestoConnectorId);
    }

    // Re-register with CuDF properties.
    auto properties = makeConnectorProperties();
    cudf_velox::connector::hive::CudfHiveConnectorFactory cudfHiveFactory;
    auto cudfHiveConnector = cudfHiveFactory.newConnector(
        prestoConnectorId, properties, ioExecutor_.get());
    connector::ConnectorRegistry::global().insert(
        cudfHiveConnector->connectorId(), cudfHiveConnector);
  }

  queryConfigs_[cudf_velox::CudfFromVelox::kGpuBatchSizeRows] =
      std::to_string(FLAGS_cudf_gpu_batch_size_rows);
}

std::shared_ptr<config::ConfigBase>
CudfTpcdsBenchmark::makeConnectorProperties() {
  auto cfg = TpcdsBenchmark::makeConnectorProperties();
  using CudfHiveCfg = cudf_velox::connector::hive::CudfHiveConfig;

  cfg->set(
      CudfHiveCfg::kMaxChunkReadLimit,
      std::to_string(FLAGS_cudf_chunk_read_limit));
  cfg->set(
      CudfHiveCfg::kMaxPassReadLimit,
      std::to_string(FLAGS_cudf_pass_read_limit));
  cfg->set(CudfHiveCfg::kAllowMismatchedCudfHiveSchemas, "true");

  return cfg;
}

void CudfTpcdsBenchmark::shutdown() {
  // CudfTpcdsQueryBuilder::shutdown() handles unregisterCudf().
  TpcdsBenchmark::shutdown();
}

void CudfTpcdsBenchmark::prepareCursorParameters(CursorParameters& params) {
  if (!queryExecutor_) {
    return;
  }
  params.queryCtx = core::QueryCtx::create(
      queryExecutor_.get(),
      core::QueryConfig({}),
      std::unordered_map<std::string, std::shared_ptr<config::ConfigBase>>{},
      cache_.get());
  cudf_velox::attachCudfMemoryPool(*params.queryCtx);
}

void CudfTpcdsBenchmark::runSequence() {
  VELOX_CHECK_GT(FLAGS_suite_passes, 0);
  VELOX_CHECK_EQ(
      FLAGS_num_repeats, 1, "Sequence measures one execution per record");
  VELOX_CHECK(!FLAGS_suite_output.empty());
  std::vector<std::string> tokens;
  folly::split(',', FLAGS_suite_queries, tokens);
  std::vector<std::pair<int32_t, TpchPlan>> plans;
  for (const auto& token : tokens) {
    const auto id = folly::to<int32_t>(token);
    plans.emplace_back(
        id, queryBuilder_->getQueryPlan(id, planDir_, pool_.get()));
  }
  std::ofstream output(FLAGS_suite_output);
  VELOX_CHECK(output.good(), "Cannot open {}", FLAGS_suite_output);
  for (int32_t pass = 0; pass < FLAGS_suite_passes; ++pass) {
    for (const auto& [id, plan] : plans) {
      // A single adaptor counts all actual allocations when output shares the
      // temporary resource. Summing peaks of distinct resources is not exact.
      VELOX_CHECK(!FLAGS_suite_memory || !cudf_velox::outputStatsMr_);
      int64_t initialBytes = 0;
      bool countersPushed = false;
      SCOPE_EXIT {
        if (countersPushed) {
          cudf_velox::statsMr_->pop_counters();
        }
      };
      if (FLAGS_suite_memory) {
        VELOX_CHECK(cudf_velox::statsMr_.has_value());
        initialBytes = cudf_velox::statsMr_->push_counters().first.value;
        countersPushed = true;
      }
      const auto start = std::chrono::steady_clock::now();
      auto [cursor, results] = run(plan, queryConfigs_);
      const auto end = std::chrono::steady_clock::now();
      VELOX_CHECK_NOT_NULL(cursor, "Query {} failed", id);
      folly::dynamic record = folly::dynamic::object("pass", pass)("query", id)(
          "task_wall_ns",
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count());
      if (FLAGS_suite_memory) {
        const auto counters = cudf_velox::statsMr_->get_bytes_counter();
        record["gpu_live_peak_bytes"] = initialBytes + counters.peak;
        record["gpu_live_start_bytes"] = initialBytes;
        record["gpu_live_end_bytes"] = initialBytes + counters.value;
      }
      folly::dynamic metrics = folly::dynamic::object();
      uint64_t rawInputBytes = 0;
      for (const auto& pipeline : cursor->task()->taskStats().pipelineStats) {
        for (const auto& op : pipeline.operatorStats) {
          rawInputBytes += op.rawInputBytes;
          for (const auto& [name, metric] : op.runtimeStats) {
            if (!metrics.count(name)) {
              metrics[name] = int64_t{0};
            }
            metrics[name] = metrics[name].asInt() + metric.sum;
          }
        }
      }
      record["raw_input_bytes"] = rawInputBytes;
      record["runtime_metrics"] = std::move(metrics);
      auto columns = folly::dynamic::array();
      for (const auto& name : plan.plan->outputType()->names()) {
        columns.push_back(name);
      }
      record["columns"] = std::move(columns);
      auto rows = folly::dynamic::array();
      for (const auto& batch : results) {
        for (vector_size_t row = 0; row < batch->size(); ++row) {
          auto values = folly::dynamic::array();
          for (const auto& column : batch->children()) {
            values.push_back(
                folly::parseJson(
                    column->variantAt(row).toJson(column->type())));
          }
          rows.push_back(std::move(values));
        }
      }
      record["rows"] = std::move(rows);
      output << folly::toJson(record) << '\n';
      output.flush();
      VELOX_CHECK(output.good(), "Cannot write {}", FLAGS_suite_output);
    }
  }
}

int main(int argc, char** argv) {
  std::string kUsage(
      "This program benchmarks TPC-DS queries with CuDF GPU acceleration.\n"
      "  --data_path        Path to TPC-DS data directory\n"
      "  --plan_path        Path to plan JSON directory\n"
      "  --cudf_properties  Path to CudfConfig properties file\n");
  gflags::SetUsageMessage(kUsage);
  folly::Init init{&argc, &argv, false};
  if (FLAGS_suite_passes > 0) {
    CudfTpcdsBenchmark benchmark;
    benchmark.initialize();
    benchmark.runSequence();
    benchmark.shutdown();
  } else {
    tpcdsBenchmark = std::make_unique<CudfTpcdsBenchmark>();
    tpcdsBenchmarkMain();
  }
}
