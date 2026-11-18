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

#include "velox/exec/Driver.h"
#include "velox/exec/Task.h"

namespace facebook::velox::cudf_velox::connector::hive {

/// An I/O worker may arbitrate the reading query's CPU staging allocation.
/// A driver draining that worker must acknowledge a task pause, including
/// during submission failure and cancellation cleanup. No-op outside drivers
/// or when an outer scope already suspended this driver.
class ScopedReadSuspension {
 public:
  ScopedReadSuspension() {
    auto* context = exec::driverThreadContext();
    if (!context) {
      return;
    }
    auto* driver = context->driverCtx()->driver;
    if (!driver->state().suspended() &&
        driver->task()->enterSuspended(
            driver->state(), /*allowTerminated=*/true) ==
            exec::StopReason::kNone) {
      driver_ = driver;
    }
    // Cleanup must drain accepted I/O even after cancellation; otherwise an
    // I/O worker arbitrating staging memory could wait for this driver forever.
  }

  // Submit/wait callers continue to their normal driver stop check, where
  // termination is observed. Cleanup callers must finish draining accepted I/O
  // even after cancellation; they do not throw from this destructor.
  ~ScopedReadSuspension() {
    if (driver_) {
      driver_->task()->leaveSuspended(driver_->state());
    }
  }

  ScopedReadSuspension(const ScopedReadSuspension&) = delete;
  ScopedReadSuspension& operator=(const ScopedReadSuspension&) = delete;

 private:
  exec::Driver* driver_{nullptr};
};

} // namespace facebook::velox::cudf_velox::connector::hive
