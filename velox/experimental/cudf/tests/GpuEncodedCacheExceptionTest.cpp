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

#include "velox/experimental/cudf/exec/CudfMemoryArbitration.h"
#include "velox/experimental/cudf/exec/GpuEncodedCache.h"

#include <rmm/cuda_stream.hpp>

#include <dlfcn.h>
#include <gtest/gtest.h>
#include <sys/prctl.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <thread>

namespace {
// Keep process-wide allocation/CUDA overrides in this dedicated executable.
// Each scenario execs a fresh child: a broken noexcept destructor is an
// observable failed test, and leaked pre-fix state cannot poison later cases.
thread_local bool failAllocation = false;
thread_local bool failWait = false;
thread_local bool failRecord = false;
thread_local bool failQuery = false;
thread_local bool injected = false;
} // namespace

void* operator new(std::size_t size) {
  if (failAllocation) {
    failAllocation = false;
    injected = true;
    throw std::bad_alloc{};
  }
  if (auto* result = std::malloc(size ? size : 1)) {
    return result;
  }
  throw std::bad_alloc{};
}

void operator delete(void* ptr) noexcept {
  std::free(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept {
  std::free(ptr);
}

extern "C" cudaError_t CUDARTAPI cudaStreamWaitEvent(
    cudaStream_t stream,
    cudaEvent_t event,
    unsigned int flags) {
  static const auto next = reinterpret_cast<decltype(&cudaStreamWaitEvent)>(
      dlsym(RTLD_NEXT, "cudaStreamWaitEvent"));
  if (failWait) {
    failWait = false;
    injected = true;
    return cudaErrorInvalidResourceHandle;
  }
  return next(stream, event, flags);
}

extern "C" cudaError_t CUDARTAPI
cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  static const auto next = reinterpret_cast<decltype(&cudaEventRecord)>(
      dlsym(RTLD_NEXT, "cudaEventRecord"));
  if (failRecord) {
    failRecord = false;
    injected = true;
    return cudaErrorMemoryAllocation;
  }
  return next(event, stream);
}

extern "C" cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) {
  static const auto next = reinterpret_cast<decltype(&cudaEventQuery)>(
      dlsym(RTLD_NEXT, "cudaEventQuery"));
  if (failQuery) {
    failQuery = false;
    injected = true;
    return cudaErrorInvalidResourceHandle;
  }
  return next(event);
}

namespace facebook::velox::cudf_velox {
namespace {
enum class Scenario {
  HitAllocation,
  HitWait,
  CreatorAllocation,
  ReleaseAllocation,
  ReleaseEvent,
  EvictionAllocation,
  EvictionEvent,
};

[[noreturn]] void runScenario(Scenario scenario) {
  prctl(PR_SET_DUMPABLE, 0);
  memory::MemoryManager::initialize({});
  setCudfContextDevice(0);
  auto resource = createCudfCustomMemoryResource(128L << 20, true);
  auto cache = std::make_shared<GpuEncodedCache>(
      resource, createMemoryResource("cuda", 0));
  rmm::cuda_stream stream;
  constexpr uint64_t bytes = 8L << 20;
  auto fill = [&](void* dst) {
    GpuEncodedCache::check(cudaMemsetAsync(dst, 42, bytes, stream.value()));
    if (scenario == Scenario::CreatorAllocation) {
      failAllocation = true;
    }
  };

  GpuEncodedCache::LeasePtr lease;
  try {
    lease = cache->getOrFill("entry", bytes, stream.value(), fill);
  } catch (const std::bad_alloc&) {
    if (scenario != Scenario::CreatorAllocation) {
      std::_Exit(10);
    }
  }
  failAllocation = false;
  if (scenario == Scenario::HitAllocation || scenario == Scenario::HitWait) {
    lease.reset();
    injected = false;
    failAllocation = scenario == Scenario::HitAllocation;
    failWait = scenario == Scenario::HitWait;
    try {
      lease = cache->get("entry", stream.value());
    } catch (const std::exception&) {
    }
    failAllocation = failWait = false;
    if (!injected) {
      std::_Exit(11);
    }
  }

  std::atomic<bool> consumed{false};
  if (scenario == Scenario::ReleaseAllocation ||
      scenario == Scenario::ReleaseEvent) {
    GpuEncodedCache::check(cudaLaunchHostFunc(
        stream.value(),
        [](void* ptr) {
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
          static_cast<std::atomic<bool>*>(ptr)->store(true);
        },
        &consumed));
    failAllocation = scenario == Scenario::ReleaseAllocation;
    failRecord = scenario == Scenario::ReleaseEvent;
  }
  lease.reset();
  failAllocation = failRecord = false;

  uint64_t freed = 0;
  failAllocation = scenario == Scenario::EvictionAllocation;
  failQuery = scenario == Scenario::EvictionEvent;
  try {
    freed = cache->evict(0);
  } catch (const std::exception&) {
  }
  failAllocation = failQuery = false;
  freed += cache->evict(0);
  if (freed != bytes || cache->pool()->usedBytes() != 0) {
    std::fprintf(
        stderr,
        "stranded allocation: freed=%lu live=%lu\n",
        freed,
        cache->pool()->usedBytes());
    std::_Exit(12);
  }
  const auto state = cache->snapshot();
  if (state["eligible_bytes"].asInt() != 0 ||
      state["pending_free_bytes"].asInt() != 0) {
    std::_Exit(13);
  }
  if ((scenario == Scenario::ReleaseAllocation ||
       scenario == Scenario::ReleaseEvent) &&
      !consumed.load()) {
    std::_Exit(14);
  }
  if ((scenario == Scenario::ReleaseEvent ||
       scenario == Scenario::EvictionEvent) &&
      !injected) {
    std::_Exit(15);
  }
  cache.reset();
  std::_Exit(0);
}

class GpuEncodedCacheExceptionTest : public testing::Test {
 protected:
  void SetUp() override {
    testing::FLAGS_gtest_death_test_style = "threadsafe";
  }
};

TEST_F(GpuEncodedCacheExceptionTest, hitAllocationRollsBackPin) {
  EXPECT_EXIT(
      runScenario(Scenario::HitAllocation), testing::ExitedWithCode(0), "");
}

TEST_F(GpuEncodedCacheExceptionTest, hitWaitFailureRollsBackPin) {
  EXPECT_EXIT(runScenario(Scenario::HitWait), testing::ExitedWithCode(0), "");
}

TEST_F(GpuEncodedCacheExceptionTest, creatorMetadataFailureRetainsOwnership) {
  EXPECT_EXIT(
      runScenario(Scenario::CreatorAllocation), testing::ExitedWithCode(0), "");
}

TEST_F(GpuEncodedCacheExceptionTest, releaseNeedsNoHostAllocation) {
  EXPECT_EXIT(
      runScenario(Scenario::ReleaseAllocation), testing::ExitedWithCode(0), "");
}

TEST_F(GpuEncodedCacheExceptionTest, releaseEventFailureWaitsForConsumer) {
  EXPECT_EXIT(
      runScenario(Scenario::ReleaseEvent), testing::ExitedWithCode(0), "");
}

TEST_F(GpuEncodedCacheExceptionTest, evictionMetadataFailureRetainsVictims) {
  EXPECT_EXIT(
      runScenario(Scenario::EvictionAllocation),
      testing::ExitedWithCode(0),
      "");
}

TEST_F(GpuEncodedCacheExceptionTest, evictionEventFailureRetainsVictims) {
  EXPECT_EXIT(
      runScenario(Scenario::EvictionEvent), testing::ExitedWithCode(0), "");
}
} // namespace
} // namespace facebook::velox::cudf_velox
