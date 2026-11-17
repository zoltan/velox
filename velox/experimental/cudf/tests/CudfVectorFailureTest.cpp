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

#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/common/memory/Memory.h"

#include <cudf/column/column_factories.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <dlfcn.h>
#include <gtest/gtest.h>

#include <cstdlib>

namespace {
thread_local bool failEvent = false;
thread_local bool failSync = false;
thread_local bool eventFailed = false;
thread_local bool syncFailed = false;
} // namespace

extern "C" cudaError_t CUDARTAPI
cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  static const auto next = reinterpret_cast<decltype(&cudaEventRecord)>(
      dlsym(RTLD_NEXT, "cudaEventRecord"));
  if (failEvent) {
    failEvent = false;
    eventFailed = true;
    return cudaErrorInvalidResourceHandle;
  }
  return next(event, stream);
}

extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
  static const auto next = reinterpret_cast<decltype(&cudaStreamSynchronize)>(
      dlsym(RTLD_NEXT, "cudaStreamSynchronize"));
  if (failSync) {
    failSync = false;
    syncFailed = true;
    return cudaErrorLaunchFailure;
  }
  return next(stream);
}

namespace facebook::velox::cudf_velox {
namespace {
// Re-exec each scenario in a fresh process. Simulated completion errors must
// neither terminate in a destructor nor release storage before a proven drain.
void releaseWithFailure(bool failDrain) {
  memory::MemoryManager::initialize({});
  auto pool = memory::memoryManager()->addLeafPool();
  rmm::cuda_stream stream;
  rmm::mr::statistics_resource_adaptor resource{
      rmm::mr::cuda_async_memory_resource{}};
  rmm::device_async_resource_ref mr{resource};
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(
      cudf::make_numeric_column(
          cudf::data_type{cudf::type_id::INT64},
          16,
          cudf::mask_state::UNALLOCATED,
          stream,
          mr));
  auto input = std::make_shared<CudfVector>(
      pool.get(),
      ROW({"c0"}, {BIGINT()}),
      16,
      std::make_unique<cudf::table>(std::move(columns)),
      stream);
  rmm::cuda_stream readerStream;
  auto reader =
      std::dynamic_pointer_cast<CudfVector>(input->shareForRead(pool.get()));
  reader->rebindStream(readerStream);
  input.reset();
  stream.synchronize();
  failEvent = true;
  failSync = failDrain;
  reader.reset();
  const bool retained = resource.get_bytes_counter().value != 0;
  std::_Exit(
      eventFailed && syncFailed == failDrain && retained == failDrain ? 0 : 1);
}

TEST(CudfVectorFailureDeathTest, eventFailureDrainsBeforeRelease) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(releaseWithFailure(false), ::testing::ExitedWithCode(0), "");
}

TEST(CudfVectorFailureDeathTest, failedDrainRetainsStorage) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  ASSERT_EXIT(releaseWithFailure(true), ::testing::ExitedWithCode(0), "");
}
} // namespace
} // namespace facebook::velox::cudf_velox
