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

#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"

#include <deque>
#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox {

class CudfFromVelox : public CudfOperatorBase {
 public:
  static constexpr const char* kGpuBatchSizeRows =
      "velox.cudf.gpu_batch_size_rows";

  CudfFromVelox(
      int32_t operatorId,
      RowTypePtr outputType,
      exec::DriverCtx* driverCtx,
      std::string planNodeId);

  /// Allows CTE scans to forward batches already in the required domain.
  /// Call before supplying input. Ordinary conversion edges stay strict.
  void allowMatchingInput() {
    allowMatchingInput_ = true;
  }

  bool needsInput() const override {
    return !finished_ && !matchingInput_;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_;
  }

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doClose() override;

 private:
  const std::optional<std::string> timestampTimeZone_;
  std::vector<RowVectorPtr> inputs_;
  std::size_t currentOutputSize_ = 0;
  // CTE producers retain their input domain; each reader normalizes locally.
  bool allowMatchingInput_{false};
  RowVectorPtr matchingInput_;
  bool finished_ = false;
};

class CudfToVelox : public CudfOperatorBase {
 public:
  static constexpr const char* kPassthroughMode =
      "velox.cudf.to_velox.passthrough_mode";

  CudfToVelox(
      int32_t operatorId,
      RowTypePtr outputType,
      exec::DriverCtx* driverCtx,
      std::string planNodeId);

  /// Allows CTE scans to forward batches already in the required domain.
  /// Call before supplying input. Ordinary conversion edges stay strict.
  void allowMatchingInput() {
    allowMatchingInput_ = true;
  }

  bool needsInput() const override {
    return !finished_ && !matchingInput_;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_;
  }

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doClose() override;

 private:
  bool isPassthroughMode() const;
  std::optional<uint64_t> averageRowSize();
  // Convert inputs_.front() to Velox once; slice it CPU-side per batch.
  RowVectorPtr convertFrontToVelox();
  std::optional<uint64_t> averageRowSize_;
  std::deque<CudfVectorPtr> inputs_;
  // Converted CPU-side buffer being drained by successive doGetOutput() calls.
  RowVectorPtr veloxBuffer_;
  // Current offset into veloxBuffer_ for the next slice.
  vector_size_t veloxOffset_{0};
  // CTE producers retain their input domain; each reader normalizes locally.
  bool allowMatchingInput_{false};
  RowVectorPtr matchingInput_;
  bool finished_ = false;
};

} // namespace facebook::velox::cudf_velox
