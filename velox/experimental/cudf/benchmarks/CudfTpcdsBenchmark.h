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

#include "velox/experimental/cudf/exec/GpuEncodedCache.h"

#include "velox/benchmarks/tpcds/TpcdsBenchmark.h"

#include <folly/executors/CPUThreadPoolExecutor.h>

#include <memory>
#include <string>
#include <vector>

/// CuDF-accelerated TPC-DS benchmark. Extends TpcdsBenchmark by:
/// - Replacing the HiveConnector with CudfHiveConnector.
/// - Registering cuDF GPU operator replacements.
/// - Adding CuDF-specific configuration flags.
class CudfTpcdsBenchmark : public TpcdsBenchmark {
 public:
  void initialize() override;

  std::shared_ptr<facebook::velox::config::ConfigBase> makeConnectorProperties()
      override;

  void shutdown() override;

  /// Run saved plans in order, retaining the process caches between queries.
  void runSequence();

 protected:
  /// Creates CudfTpcdsQueryBuilder and calls enableCudf().
  void initQueryBuilder() override;

  void prepareCursorParameters(
      facebook::velox::exec::CursorParameters& params) override;

 private:
  // Outlives all query contexts created by the sequence harness.
  std::unique_ptr<folly::CPUThreadPoolExecutor> queryExecutor_;
  // Shares the GPU budget with query operators and outlives all input leases.
  std::shared_ptr<facebook::velox::cudf_velox::GpuEncodedCache> deviceCache_;
};
