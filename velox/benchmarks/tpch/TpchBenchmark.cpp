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

#include "velox/benchmarks/tpch/TpchBenchmark.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include "velox/exec/OperatorType.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/QueryAssertions.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::dwio::common;

DEFINE_string(
    data_path,
    "",
    "Root path of TPC-H data. Data layout must follow Hive-style partitioning. "
    "Example layout for '-data_path=/data/tpch10'\n"
    "       /data/tpch10/customer\n"
    "       /data/tpch10/lineitem\n"
    "       /data/tpch10/nation\n"
    "       /data/tpch10/orders\n"
    "       /data/tpch10/part\n"
    "       /data/tpch10/partsupp\n"
    "       /data/tpch10/region\n"
    "       /data/tpch10/supplier\n"
    "If the above are directories, they contain the data files for "
    "each table. If they are files, they contain a file system path for each "
    "data file, one per line. This allows running against cloud storage or "
    "HDFS");
namespace {
static bool notEmpty(const char* /*flagName*/, const std::string& value) {
  return !value.empty();
}
} // namespace

DEFINE_validator(data_path, &notEmpty);

DEFINE_int32(
    run_query_verbose,
    -1,
    "Run a given query and print execution statistics");
DEFINE_int32(
    io_meter_column_pct,
    0,
    "Percentage of lineitem columns to "
    "include in IO meter query. The columns are sorted by name and the n% first "
    "are scanned");

DECLARE_int32(num_drivers);
DECLARE_int32(num_repeats);
DEFINE_bool(
    cte_compare,
    false,
    "Validate and measure original, duplicated and CTE plans");
DEFINE_int32(
    cte_samples,
    3,
    "Measured CTE comparison samples after one warmup");
DEFINE_uint64(
    cte_max_retained_mb,
    0,
    "Optional per-CTE retained-byte limit in MiB (0 means unlimited)");
DEFINE_string(
    cte_reference_path,
    "",
    "Directory of qN.parquet reference results; empty uses original plan");
DEFINE_string(
    cte_plan_directory,
    "",
    "Optional directory for comparison plans and execution statistics");

namespace {
CteOptions cteOptions() {
  CteOptions options;
  if (FLAGS_cte_max_retained_mb != 0) {
    VELOX_USER_CHECK_LE(
        FLAGS_cte_max_retained_mb, std::numeric_limits<uint64_t>::max() >> 20);
    options.maxRetainedBytes = FLAGS_cte_max_retained_mb << 20;
  }
  return options;
}
} // namespace

void TpchBenchmark::initQueryBuilder() {
  queryBuilder_ =
      std::make_shared<TpchQueryBuilder>(toFileFormat(FLAGS_data_format));
  queryBuilder_->initialize(FLAGS_data_path);
}

void TpchBenchmark::initialize() {
  QueryBenchmarkBase::initialize();
  initQueryBuilder();
}

void TpchBenchmark::shutdown() {
  QueryBenchmarkBase::shutdown();
  queryBuilder_.reset();
}

void TpchBenchmark::runMain(
    std::ostream& out,
    facebook::velox::RunStats& runStats) {
  if (FLAGS_cte_compare) {
    compareCtePlans(out);
    return;
  }
  if (FLAGS_run_query_verbose == -1 && FLAGS_io_meter_column_pct == 0) {
    folly::runBenchmarks();
  } else {
    auto queryPlan = FLAGS_io_meter_column_pct > 0
        ? queryBuilder_->getIoMeterPlan(FLAGS_io_meter_column_pct)
        : queryBuilder_->getQueryPlan(FLAGS_run_query_verbose);
    auto [cursor, actualResults] = run(queryPlan, queryConfigs_);
    if (!cursor) {
      LOG(ERROR) << "Query terminated with error. Exiting";
      exit(1);
    }
    auto task = cursor->task();
    ensureTaskCompletion(task.get());
    if (FLAGS_include_results) {
      printResults(actualResults, out);
      out << std::endl;
    }
    const auto stats = task->taskStats();
    int64_t rawInputBytes = 0;
    for (auto& pipeline : stats.pipelineStats) {
      auto& first = pipeline.operatorStats[0];
      if (first.operatorType == OperatorType::kTableScan) {
        rawInputBytes += first.rawInputBytes;
      }
    }
    runStats.rawInputBytes = rawInputBytes;
    out << fmt::format(
               "Execution time: {}",
               facebook::velox::succinctMillis(
                   stats.executionEndTimeMs - stats.executionStartTimeMs))
        << std::endl;
    out << fmt::format(
               "Splits total: {}, finished: {}",
               stats.numTotalSplits,
               stats.numFinishedSplits)
        << std::endl;
    out << printPlanWithStats(
               *queryPlan.plan, stats, FLAGS_include_custom_stats)
        << std::endl;
  }
}

void TpchBenchmark::compareCtePlans(std::ostream& out) {
  VELOX_USER_CHECK_GE(FLAGS_cte_samples, 1);
  VELOX_USER_CHECK_EQ(
      FLAGS_num_repeats, 1, "CTE comparison manages its own repetitions");
  const auto queries = FLAGS_run_query_verbose == -1
      ? std::vector<int>{2, 11, 15, 17, 20, 22}
      : std::vector<int>{FLAGS_run_query_verbose};
  for (auto query : queries) {
    const auto options = cteOptions();
    const std::vector<std::pair<std::string, TpchPlan>> plans{
        {"original", queryBuilder_->getQueryPlan(query)},
        {"duplicated", queryBuilder_->getCtePlan(query, false, options)},
        {"cte", queryBuilder_->getCtePlan(query, true, options)}};
    auto referencePlan = plans[0].second;
    if (!FLAGS_cte_reference_path.empty()) {
      auto node = PlanBuilder()
                      .tableScan(plans[2].second.plan->outputType())
                      .planNode();
      referencePlan = TpchPlan{
          node,
          {{node->id(),
            {fmt::format("{}/q{}.parquet", FLAGS_cte_reference_path, query)}}},
          FileFormat::PARQUET};
    }
    auto reference = run(referencePlan, queryConfigs_);
    VELOX_CHECK_NOT_NULL(
        reference.first, "Reference execution failed for Q{}", query);
    for (int sample = -1; sample < FLAGS_cte_samples; ++sample) {
      // Rotate order after warmup to reduce consistent ordering bias.
      for (int index = 0; index < plans.size(); ++index) {
        const auto& [name, plan] = plans[(index + sample + 1) % plans.size()];
        const auto start = std::chrono::steady_clock::now();
        auto result = run(plan, queryConfigs_);
        const auto elapsed = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
        VELOX_CHECK_NOT_NULL(result.first, "Q{} {} failed", query, name);
        VELOX_CHECK(
            assertEqualResults(reference.second, result.second),
            "Q{} {} result differs from reference",
            query,
            name);
        const auto task = result.first->task();
        const auto stats = task->taskStats();
        uint64_t cpu = 0, cteCpu = 0, allocations = 0, scanRows = 0,
                 producers = 0, readers = 0, retainedBytes = 0;
        for (const auto& pipeline : stats.pipelineStats) {
          for (const auto& op : pipeline.operatorStats) {
            auto nanos = op.addInputTiming.cpuNanos +
                op.getOutputTiming.cpuNanos + op.finishTiming.cpuNanos +
                op.isBlockedTiming.cpuNanos;
            cpu += nanos;
            allocations += op.memoryStats.numMemoryAllocations;
            if (op.operatorType == "SharedScan" ||
                op.operatorType == "SharedResultSink") {
              cteCpu += nanos;
            }
            if (op.operatorType == OperatorType::kTableScan) {
              scanRows += op.rawInputPositions;
            }
            for (const auto& [key, counter] : op.runtimeStats) {
              if (key == "activeProducerDrivers") {
                producers += counter.sum;
              } else if (key == "activeReaderDrivers") {
                readers += counter.sum;
              } else if (key == "peakRetainedBytes") {
                retainedBytes += counter.sum;
              }
            }
          }
        }
        if (!FLAGS_cte_plan_directory.empty() && sample == -1) {
          std::filesystem::create_directories(FLAGS_cte_plan_directory);
          std::ofstream file(fmt::format(
              "{}/q{}-{}-d{}.txt",
              FLAGS_cte_plan_directory,
              query,
              name,
              FLAGS_num_drivers));
          VELOX_CHECK(file.good());
          file << printPlanWithStats(*plan.plan, stats, true);
        }
        out << fmt::format(
            "CTE_TPCH query={} variant={} drivers={} sample={} seconds={:.6f} "
            "cpu_seconds={:.6f} cte_cpu_seconds={:.6f} peak_bytes={} "
            "peak_retained_bytes={} pool_allocations={} scan_rows={} active_producers={} active_readers={} validated=true\n",
            query,
            name,
            FLAGS_num_drivers,
            sample,
            elapsed,
            cpu / 1e9,
            cteCpu / 1e9,
            task->pool()->stats().peakBytes,
            retainedBytes,
            allocations,
            scanRows,
            producers,
            readers);
        out.flush();
      }
    }
  }
}

std::unique_ptr<TpchBenchmark> benchmark;

BENCHMARK(q1) {
  benchmark->runQuery(1);
}

BENCHMARK(q2) {
  benchmark->runQuery(2);
}

BENCHMARK(q3) {
  benchmark->runQuery(3);
}

BENCHMARK(q4) {
  benchmark->runQuery(4);
}

BENCHMARK(q5) {
  benchmark->runQuery(5);
}

BENCHMARK(q6) {
  benchmark->runQuery(6);
}

BENCHMARK(q7) {
  benchmark->runQuery(7);
}

BENCHMARK(q8) {
  benchmark->runQuery(8);
}

BENCHMARK(q9) {
  benchmark->runQuery(9);
}

BENCHMARK(q10) {
  benchmark->runQuery(10);
}

BENCHMARK(q11) {
  benchmark->runQuery(11);
}

BENCHMARK(q12) {
  benchmark->runQuery(12);
}

BENCHMARK(q13) {
  benchmark->runQuery(13);
}

BENCHMARK(q14) {
  benchmark->runQuery(14);
}

BENCHMARK(q15) {
  benchmark->runQuery(15);
}

BENCHMARK(q16) {
  benchmark->runQuery(16);
}

BENCHMARK(q17) {
  benchmark->runQuery(17);
}

BENCHMARK(q18) {
  benchmark->runQuery(18);
}

BENCHMARK(q19) {
  benchmark->runQuery(19);
}

BENCHMARK(q20) {
  benchmark->runQuery(20);
}

BENCHMARK(q21) {
  benchmark->runQuery(21);
}

BENCHMARK(q22) {
  benchmark->runQuery(22);
}

BENCHMARK(q2_cte) {
  benchmark->runCteQuery(2, cteOptions());
}

BENCHMARK(q11_cte) {
  benchmark->runCteQuery(11, cteOptions());
}

BENCHMARK(q15_cte) {
  benchmark->runCteQuery(15, cteOptions());
}

BENCHMARK(q17_cte) {
  benchmark->runCteQuery(17, cteOptions());
}

BENCHMARK(q20_cte) {
  benchmark->runCteQuery(20, cteOptions());
}

BENCHMARK(q22_cte) {
  benchmark->runCteQuery(22, cteOptions());
}

void tpchBenchmarkMain() {
  VELOX_CHECK_NOT_NULL(benchmark);
  benchmark->initialize();
  if (FLAGS_test_flags_file.empty()) {
    RunStats ignore;
    benchmark->runMain(std::cout, ignore);
  } else {
    benchmark->runAllCombinations();
  }
  benchmark->shutdown();
}
