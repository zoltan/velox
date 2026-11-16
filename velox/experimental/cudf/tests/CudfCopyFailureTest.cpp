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

// Isolated cuDF copy shim: report failure after accepting a real copy. Retain
// actual source and destination allocations through test cleanup so premature
// logical release is observed without making the test itself access freed data.
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet_io_utils.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <vector>

namespace {
struct CopyState {
  std::promise<void> submitted;
  std::promise<void> release;
  std::shared_future<void> gate = release.get_future().share();
  std::atomic<bool> complete{false};
  std::atomic<bool> prematureRelease{false};
  std::atomic<bool> injected{false};
};
thread_local CopyState* injectCopyFailure = nullptr;
} // namespace

namespace cudf::detail {
cudaError_t memcpy_batch_async(
    void* const* destinations,
    const void* const* sources,
    const size_t* sizes,
    size_t count,
    cuda::stream_ref stream) {
  // Interpose the exported cuDF helper, since libcudf may hide a static cudart.
  // Use real CUDA copies, just as the helper's sequential fallback does.
  auto* state = injectCopyFailure;
  injectCopyFailure = nullptr;
  auto status = cudaSuccess;
  if (state) {
    status = cudaLaunchHostFunc(
        stream.get(),
        [](void* data) { static_cast<CopyState*>(data)->gate.wait(); },
        state);
  }
  for (size_t i = 0; status == cudaSuccess && i < count; ++i) {
    status = cudaMemcpyAsync(
        destinations[i], sources[i], sizes[i], cudaMemcpyDefault, stream.get());
  }
  if (!state) {
    return status;
  }
  if (status == cudaSuccess) {
    status = cudaLaunchHostFunc(
        stream.get(),
        [](void* data) { static_cast<CopyState*>(data)->complete = true; },
        state);
  }
  state->injected = status == cudaSuccess;
  state->submitted.set_value();
  return status == cudaSuccess ? cudaErrorInvalidValue : status;
}
} // namespace cudf::detail

namespace {
class ObservedBuffer : public cudf::io::datasource::buffer {
 public:
  ObservedBuffer(const uint8_t* bytes, CopyState* state)
      : bytes_(bytes), state_(state) {}
  ~ObservedBuffer() override {
    if (!state_->complete) {
      state_->prematureRelease = true;
    }
  }
  size_t size() const override {
    return 64;
  }
  const uint8_t* data() const override {
    return bytes_;
  }

 private:
  const uint8_t* bytes_;
  CopyState* state_;
};

class HostSource : public cudf::io::datasource {
 public:
  HostSource(const uint8_t* bytes, CopyState* state)
      : bytes_(bytes), state_(state) {}
  size_t size() const override {
    return 64;
  }
  std::unique_ptr<buffer> host_read(size_t, size_t) override {
    return std::make_unique<ObservedBuffer>(bytes_, state_);
  }
  size_t host_read(size_t, size_t, uint8_t*) override {
    throw std::logic_error("Unexpected destination-filling read");
  }

 private:
  const uint8_t* bytes_;
  CopyState* state_;
};

// Defer actual device frees to the test after it has joined the accepted copy.
struct RetainedAllocations {
  struct Allocation {
    void* pointer;
    size_t bytes;
    size_t alignment;
  };
  std::array<Allocation, 4> freed{};
  size_t count{0};
  rmm::mr::cuda_memory_resource upstream;
};
struct RetainingResource {
  RetainedAllocations* state;
  void* allocate(cuda::stream_ref stream, size_t bytes, size_t alignment) {
    return state->upstream.allocate(stream, bytes, alignment);
  }
  void* allocate_sync(size_t bytes, size_t alignment) {
    return state->upstream.allocate_sync(bytes, alignment);
  }
  void deallocate(
      cuda::stream_ref,
      void* pointer,
      size_t bytes,
      size_t alignment) noexcept {
    state->freed[state->count++] = {pointer, bytes, alignment};
  }
  void deallocate_sync(void* pointer, size_t bytes, size_t alignment) noexcept {
    state->freed[state->count++] = {pointer, bytes, alignment};
  }
  bool operator==(const RetainingResource&) const = default;
};
void get_property(
    const RetainingResource&,
    cuda::mr::device_accessible) noexcept {}

TEST(CudfCopyFailureTest, acceptedCopyCompletesBeforeSourceRelease) {
  rmm::cuda_stream stream;
  CopyState state;
  auto submitted = state.submitted.get_future();
  uint8_t* bytes = nullptr;
  ASSERT_EQ(
      cudaHostAlloc(reinterpret_cast<void**>(&bytes), 64, cudaHostAllocDefault),
      cudaSuccess);
  std::fill(bytes, bytes + 64, 42);
  HostSource source(bytes, &state);
  RetainedAllocations allocations;
  RetainingResource resource{&allocations};
  std::promise<bool> returned;
  auto result = returned.get_future();
  std::thread caller([&] {
    injectCopyFailure = &state;
    try {
      rmm::device_async_resource_ref mr{resource};
      const std::vector<cudf::io::text::byte_range_info> ranges{{0, 64}};
      auto [buffers, views, complete] =
          cudf::io::parquet::fetch_byte_ranges_to_device_async(
              source,
              ranges,
              cudf::io::parquet::io_submission_policy::INTERLEAVE,
              stream,
              {mr, mr});
      complete.get();
      returned.set_value(false);
    } catch (...) {
      returned.set_value(true);
    }
    injectCopyFailure = nullptr;
  });
  const auto ready = submitted.wait_for(std::chrono::seconds(5));
  // Let an unguarded helper unwind; a guarded helper waits on this gate.
  result.wait_for(std::chrono::milliseconds(100));
  state.release.set_value();
  caller.join();
  stream.synchronize();
  for (size_t i = 0; i < allocations.count; ++i) {
    const auto& allocation = allocations.freed[i];
    allocations.upstream.deallocate(
        stream, allocation.pointer, allocation.bytes, allocation.alignment);
  }
  EXPECT_EQ(cudaFreeHost(bytes), cudaSuccess);
  ASSERT_EQ(ready, std::future_status::ready);
  EXPECT_TRUE(state.injected);
  EXPECT_TRUE(result.get());
  EXPECT_TRUE(state.complete);
  EXPECT_FALSE(state.prematureRelease);
}
} // namespace
