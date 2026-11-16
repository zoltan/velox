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

#include "velox/experimental/cudf/connectors/hive/BufferedInputDataSource.h"
#include "velox/experimental/cudf/exec/GpuResources.h"

#include "velox/common/file/File.h"
#include "velox/common/memory/Memory.h"

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <folly/Executor.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <future>
#include <new>
#include <string>

namespace {
thread_local bool failNextAllocation = false;
} // namespace

// Isolated executable: inject only after a test-controlled accepted submission.
void* operator new(std::size_t size) {
  if (failNextAllocation) {
    failNextAllocation = false;
    throw std::bad_alloc{};
  }
  if (auto* pointer = std::malloc(size ? size : 1)) {
    return pointer;
  }
  throw std::bad_alloc{};
}
void operator delete(void* pointer) noexcept {
  std::free(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept {
  std::free(pointer);
}

namespace facebook::velox::cudf_velox::connector::hive {
namespace {
class ControlledExecutor : public folly::Executor {
 public:
  void add(folly::Func task) override {
    task_ = std::move(task);
    // The executor now owns the task. A later allocation must not abandon it.
    failNextAllocation = true;
  }
  void drain() {
    if (task_) {
      auto task = std::move(task_);
      task();
    }
  }

 private:
  folly::Func task_;
};

class TestInput : public dwio::common::BufferedInput {
 public:
  TestInput(
      std::shared_ptr<ReadFile> file,
      memory::MemoryPool& pool,
      folly::Executor* executor)
      : BufferedInput(std::move(file), pool), executor_(executor) {}
  folly::Executor* executor() const override {
    return executor_;
  }

 private:
  folly::Executor* executor_;
};

class RecordingSource : public BufferedInputDataSource {
 public:
  using BufferedInputDataSource::BufferedInputDataSource;
  std::unique_ptr<buffer> host_read(size_t offset, size_t bytes) override {
    readCompleted = true;
    return BufferedInputDataSource::host_read(offset, bytes);
  }
  bool readCompleted{false};
};

class BufferedReadFailureTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    setCudfContextDevice(0);
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }
  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool();
    input_ = std::make_shared<TestInput>(
        std::make_shared<InMemoryReadFile>(content_), *pool_, &executor_);
    failNextAllocation = false;
  }
  void TearDown() override {
    failNextAllocation = false;
    executor_.drain();
  }
  std::string content_ = std::string(256, 'x');
  ControlledExecutor executor_;
  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<TestInput> input_;
};

TEST_F(BufferedReadFailureTest, acceptedTaskSurvivesFutureAllocationFailure) {
  rmm::cuda_stream stream;
  rmm::mr::cuda_memory_resource resource;
  rmm::device_async_resource_ref mr{resource};
  rmm::device_buffer destination(64, stream, mr);
  auto source = std::make_shared<RecordingSource>(input_);
  std::future<size_t> pending;
  bool threw = false;
  try {
    pending = source->device_read_async(
        0, 64, static_cast<uint8_t*>(destination.data()), stream);
  } catch (const std::bad_alloc&) {
    threw = true;
  }
  failNextAllocation = false;
  // An implementation may avoid the post-enqueue allocation altogether.
  EXPECT_TRUE(!threw || source->readCompleted);
  if (pending.valid()) {
    EXPECT_EQ(pending.get(), 64);
  }
  // Keep source and destination alive while draining the deliberately broken
  // old path, so the regression reports abandonment without performing a UAF.
  executor_.drain();
  EXPECT_TRUE(source->readCompleted);
}

} // namespace
} // namespace facebook::velox::cudf_velox::connector::hive
