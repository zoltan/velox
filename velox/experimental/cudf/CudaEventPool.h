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

#include <cudf/utilities/error.hpp>

#include <cuda_runtime_api.h>

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox {

/// A submitting thread's bounded pool of event handles. Ownership can cross
/// threads; the last handle retains the pool after its submitting thread exits.
class CudaEventPool {
 public:
  explicit CudaEventPool(int device) : device_(device) {
    events_.reserve(kMaxIdle);
  }

  ~CudaEventPool() {
    int previous;
    if (cudaGetDevice(&previous) != cudaSuccess ||
        cudaSetDevice(device_) != cudaSuccess) {
      return;
    }
    for (auto event : events_) {
      cudaEventDestroy(event);
    }
    cudaSetDevice(previous);
  }

  static std::shared_ptr<CudaEventPool> get() {
    static thread_local std::shared_ptr<CudaEventPool> pool;
    int device;
    CUDF_CUDA_TRY(cudaGetDevice(&device));
    if (!pool || pool->device_ != device) {
      pool = std::make_shared<CudaEventPool>(device);
    }
    return pool;
  }

  cudaEvent_t acquire() {
    {
      std::lock_guard lock(mutex_);
      if (!events_.empty()) {
        auto event = events_.back();
        events_.pop_back();
        return event;
      }
    }
    cudaEvent_t event;
    CUDF_CUDA_TRY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return event;
  }

  /// The caller must have submitted every wait/query using this recording.
  /// cudaStreamWaitEvent captures the current recording and is unaffected by
  /// later cudaEventRecord calls (CUDA Runtime API, Event Management).
  void release(cudaEvent_t event) noexcept {
    std::lock_guard lock(mutex_);
    if (events_.size() < kMaxIdle) {
      events_.push_back(event);
    } else {
      cudaEventDestroy(event);
    }
  }

 private:
  static constexpr size_t kMaxIdle = 64;
  const int device_;
  std::mutex mutex_;
  std::vector<cudaEvent_t> events_;
};

/// Exclusive event handle. Returning it to the pool permits a new recording;
/// no caller may issue further waits/queries for its old recording afterwards.
class PooledCudaEvent {
 public:
  PooledCudaEvent() : pool_(CudaEventPool::get()), event_(pool_->acquire()) {}
  PooledCudaEvent(const PooledCudaEvent&) = delete;
  PooledCudaEvent& operator=(const PooledCudaEvent&) = delete;
  PooledCudaEvent(PooledCudaEvent&& other) noexcept
      : pool_(std::move(other.pool_)),
        event_(std::exchange(other.event_, nullptr)) {}
  ~PooledCudaEvent() {
    if (event_) {
      pool_->release(event_);
    }
  }
  cudaEvent_t get() const {
    return event_;
  }

 private:
  std::shared_ptr<CudaEventPool> pool_;
  cudaEvent_t event_;
};

} // namespace facebook::velox::cudf_velox
