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

// This regression uses only public cuDF APIs so it can accompany an upstream
// helper fix. Pending reads never access their destination: a broken lifetime
// becomes an assertion failure, not a use-after-free in the test itself.
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet_io_utils.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
class GatedSource : public cudf::io::datasource {
 public:
  explicit GatedSource(bool throwDuringSubmission)
      : throwDuringSubmission_(throwDuringSubmission) {}

  size_t size() const override {
    return 256;
  }
  bool supports_device_read() const override {
    return true;
  }
  bool is_device_read_preferred(size_t) const override {
    return true;
  }
  std::unique_ptr<buffer> host_read(size_t, size_t) override {
    throw std::logic_error("Unexpected host read");
  }
  size_t host_read(size_t, size_t, uint8_t*) override {
    throw std::logic_error("Unexpected host read");
  }
  std::future<size_t>
  device_read_async(size_t, size_t, uint8_t*, cuda::stream_ref) override {
    if (++submissions_ == 1) {
      if (throwDuringSubmission_) {
        return delayed_.get_future();
      }
      std::promise<size_t> failed;
      failed.set_exception(
          std::make_exception_ptr(std::runtime_error("read failed")));
      return failed.get_future();
    }
    auto pending =
        throwDuringSubmission_ ? std::future<size_t>{} : delayed_.get_future();
    submitted_.set_value();
    if (throwDuringSubmission_) {
      throw std::runtime_error("submission failed");
    }
    return pending;
  }

  std::promise<void> submitted_;
  std::promise<size_t> delayed_;

 private:
  const bool throwDuringSubmission_;
  int submissions_{0};
};

class GatedHostSource : public cudf::io::datasource {
 public:
  size_t size() const override {
    return 256;
  }
  std::unique_ptr<buffer> host_read(size_t offset, size_t bytes) override {
    if (offset == 0) {
      throw std::runtime_error("host read failed");
    }
    submitted_.set_value();
    delayed_.get_future().get();
    return buffer::create(std::vector<uint8_t>(bytes));
  }
  size_t host_read(size_t, size_t, uint8_t*) override {
    throw std::logic_error("Unexpected destination-filling host read");
  }
  std::promise<void> submitted_;
  std::promise<size_t> delayed_;
};

template <typename Source>
void checkFailureDrains(
    Source& source,
    const std::string& expectedError,
    bool failDelayed) {
  auto submitted = source.submitted_.get_future();
  std::promise<std::string> returned;
  auto result = returned.get_future();
  std::thread caller([&] {
    try {
      rmm::cuda_stream stream;
      rmm::mr::cuda_memory_resource resource;
      rmm::device_async_resource_ref mr{resource};
      const std::vector<cudf::io::text::byte_range_info> ranges{
          {0, 64}, {128, 64}};
      auto [buffers, views, complete] =
          cudf::io::parquet::fetch_byte_ranges_to_device_async(
              source,
              ranges,
              cudf::io::parquet::io_submission_policy::INTERLEAVE,
              stream,
              {mr, mr});
      complete.get();
      returned.set_value("no error");
    } catch (const std::runtime_error& error) {
      // All helper/returned buffer destructors have run at this boundary.
      returned.set_value(error.what());
    } catch (...) {
      returned.set_exception(std::current_exception());
    }
  });
  // Always unblock and join, even if setup failed; no assertion owns a thread.
  const auto ready = submitted.wait_for(std::chrono::seconds(5));
  const bool returnedBeforeCompletion =
      result.wait_for(std::chrono::milliseconds(100)) ==
      std::future_status::ready;
  if (failDelayed) {
    source.delayed_.set_exception(
        std::make_exception_ptr(std::runtime_error("later read failed")));
  } else {
    source.delayed_.set_value(64);
  }
  caller.join();
  ASSERT_EQ(ready, std::future_status::ready);
  EXPECT_EQ(result.get(), expectedError);
  EXPECT_FALSE(returnedBeforeCompletion);
}

TEST(CudfReadFailureTest, drainsWhenLaterSubmissionThrows) {
  GatedSource source(true);
  checkFailureDrains(source, "submission failed", false);
}

TEST(CudfReadFailureTest, drainsRemainingReadsWhenFirstCompletionFails) {
  GatedSource source(false);
  checkFailureDrains(source, "read failed", true);
}

TEST(CudfReadFailureTest, drainsHostReadsWhenFirstCompletionFails) {
  GatedHostSource source;
  checkFailureDrains(source, "host read failed", true);
}
} // namespace
