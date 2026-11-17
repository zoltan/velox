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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/connectors/hive/BufferedInputDataSource.h"
#include "velox/experimental/cudf/connectors/hive/CudfSplitReaderByteFetch.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/io/CachedDeviceRead.h"

#include "velox/common/base/Exceptions.h"

#include <folly/Executor.h>

#include <algorithm>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox::connector::hive {

namespace {

// Owns host buffers until the uploads recorded on this stream complete. The
// normal path retires this object on the IO executor. Executor rejection falls
// back to draining here, preserving the same lifetime guarantee.
class HostUploadBuffers {
 public:
  explicit HostUploadBuffers(cuda::stream_ref stream) : stream_(stream) {
    ensureCudaContextForThread();
    CUDF_CUDA_TRY(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  }

  ~HostUploadBuffers() noexcept {
    try {
      ensureCudaContextForThread();
      const auto status = recorded_ ? cudaEventSynchronize(event_)
                                    : cudaStreamSynchronize(stream_.get());
      if (status != cudaSuccess) {
        // A failed context cannot establish safe DMA completion. Retain the
        // buffers rather than freeing memory that the device may still use.
        LOG(ERROR) << "Cannot retire cached input uploads: "
                   << cudaGetErrorString(status);
        retainBuffers();
      }
    } catch (...) {
      retainBuffers();
    }
    cudaEventDestroy(event_);
  }

  void recordCompletion() {
    ensureCudaContextForThread();
    CUDF_CUDA_TRY(cudaEventRecord(event_, stream_.get()));
    recorded_ = true;
  }

  std::vector<std::shared_ptr<std::unique_ptr<cudf::io::datasource::buffer>>>
      buffers;

 private:
  void retainBuffers() noexcept {
    for (auto& buffer : buffers) {
      buffer->release();
    }
  }

  cuda::stream_ref stream_;
  cudaEvent_t event_{};
  bool recorded_{false};
};

// Launches the `read` task on the IO executor or waiting thread, whichever
// starts first.
std::future<size_t> submitDeviceRead(
    folly::Executor* executor,
    std::function<size_t()> read) {
  auto task = std::make_shared<std::packaged_task<size_t()>>(std::move(read));
  auto result = task->get_future();
  auto once = std::make_shared<std::once_flag>();
  auto run = [once, task]() { std::call_once(*once, [task]() { (*task)(); }); };
  // Complete fallible future construction before the executor can accept work.
  auto completion = std::async(
      std::launch::deferred, [run, result = std::move(result)]() mutable {
        run();
        return result.get();
      });
  executor->add(std::move(run));
  return completion;
}

} // namespace

BufferedInputDataSource::BufferedInputDataSource(
    std::shared_ptr<facebook::velox::dwio::common::BufferedInput> input,
    std::shared_ptr<IoStats> ioStats,
    std::shared_ptr<const StringIdLease> fileIdentity)
    : input_(std::move(input)),
      fileSize_(input_->getReadFile()->size()),
      ioStats_(std::move(ioStats)),
      fileIdentity_(std::move(fileIdentity)),
      deviceCache_(
          input_->hasCache() && fileIdentity_ ? encodedInputGpuCache()
                                              : nullptr) {}

void BufferedInputDataSource::recordBytes(const char* name, uint64_t bytes) {
  if (ioStats_) {
    ioStats_->addCounter(
        name, RuntimeCounter(bytes, RuntimeCounter::Unit::kBytes));
  }
}

size_t BufferedInputDataSource::size() const {
  return fileSize_;
}

std::unique_ptr<cudf::io::datasource::buffer>
BufferedInputDataSource::host_read(size_t offset, size_t size) {
  if (offset >= fileSize_) {
    return cudf::io::datasource::buffer::create(std::vector<uint8_t>{});
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  std::vector<uint8_t> data(readSize);
  readContiguous(offset, readSize, data.data());
  return cudf::io::datasource::buffer::create(std::move(data));
}

size_t
BufferedInputDataSource::host_read(size_t offset, size_t size, uint8_t* dst) {
  if (offset >= fileSize_) {
    return 0;
  }
  const size_t readSize = std::min(size, fileSize_ - offset);
  readContiguous(offset, readSize, dst);
  return readSize;
}

std::future<std::unique_ptr<cudf::io::datasource::buffer>>
BufferedInputDataSource::host_read_async(size_t offset, size_t size) {
  return std::async(std::launch::deferred, [this, offset, size]() {
    return this->host_read(offset, size);
  });
}

std::future<size_t> BufferedInputDataSource::host_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  return std::async(std::launch::deferred, [this, offset, size, dst]() {
    return this->host_read(offset, size, dst);
  });
}

std::future<size_t> BufferedInputDataSource::device_read_async(
    size_t offset,
    size_t size,
    uint8_t* dst,
    cuda::stream_ref stream) {
  VELOX_CHECK(input_->executor() != nullptr, "IO executor is not initialized");
  return submitDeviceRead(
      input_->executor(), [this, offset, size, dst, stream]() {
        auto hostBuffer = host_read(offset, size);
        ensureCudaContextForThread();
        CUDF_CUDA_TRY(cudaMemcpyAsync(
            dst,
            hostBuffer->data(),
            hostBuffer->size(),
            cudaMemcpyDefault,
            stream.get()));
        recordBytes("cudfInputH2DBytes", hostBuffer->size());
        return hostBuffer->size();
      });
}

gpu::DeviceReadBatch BufferedInputDataSource::readDeviceRangesAsync(
    std::span<const gpu::DeviceByteRange> ranges,
    bool serializeSubmissions,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref outputResource,
    rmm::device_async_resource_ref temporaryResource) {
  if (deviceCache_) {
    return readCachedRanges(
        ranges,
        serializeSubmissions,
        stream,
        outputResource,
        temporaryResource);
  }
  std::vector<cudf::io::text::byte_range_info> byteRanges;
  byteRanges.reserve(ranges.size());
  for (const auto& range : ranges) {
    byteRanges.emplace_back(range.offset(), range.size());
  }
  return copyByteRangesAsync(
      shared_from_this(),
      {byteRanges.data(), byteRanges.size()},
      serializeSubmissions,
      stream,
      {outputResource, temporaryResource});
}

gpu::DeviceReadBatch BufferedInputDataSource::readCachedRanges(
    std::span<const gpu::DeviceByteRange> ranges,
    bool serializeSubmissions,
    cuda::stream_ref stream,
    rmm::device_async_resource_ref outputResource,
    rmm::device_async_resource_ref temporaryResource) {
  struct Owner {
    std::shared_ptr<BufferedInputDataSource> source;
    std::vector<GpuEncodedCache::LeasePtr> leases;
    std::vector<bool> created;
    std::vector<std::future<size_t>> pending;
    std::vector<size_t> expected;
    std::shared_ptr<HostUploadBuffers> uploads;
    folly::Executor* executor;
    cuda::stream_ref stream;
    explicit Owner(cuda::stream_ref value) : stream(value) {}
    gpu::DeviceReadBatch bypass;
    std::exception_ptr error;
    bool finished{false};

    void finish() {
      if (finished) {
        return;
      }
      finished = true;
      // Drain every accepted read before releasing any destination, preserving
      // the first failure even if a later read also fails.
      for (size_t i = 0; i < pending.size(); ++i) {
        try {
          VELOX_CHECK_EQ(
              pending[i].get(), expected[i], "Short cached input read");
        } catch (...) {
          if (!error) {
            error = std::current_exception();
          }
        }
      }
      try {
        bypass.wait();
      } catch (...) {
        if (!error) {
          error = std::current_exception();
        }
      }
      if (uploads) {
        try {
          uploads->recordCompletion();
        } catch (...) {
          if (!error) {
            error = std::current_exception();
          }
        }
        try {
          // No driver-side device wait. A rejected submission destroys the
          // capture here and drains safely before propagating read failures.
          executor->add([retirement = std::move(uploads)] {});
        } catch (...) {
          // Cleanup was synchronous; executor shutdown does not invalidate an
          // otherwise completed read.
        }
      }
      if (error) {
        for (size_t i = 0; i < leases.size(); ++i) {
          if (created[i]) {
            leases[i]->failFill();
          }
        }
      }
      if (error) {
        std::rethrow_exception(error);
      }
      for (auto& lease : leases) {
        if (lease) {
          lease->waitForFill();
        }
      }
    }

    ~Owner() {
      try {
        finish();
      } catch (...) {
        // Explicit completion reports failures. Abandonment still drains reads.
      }
    }
  };
  auto owner = std::make_shared<Owner>(stream);
  owner->source = shared_from_this();
  owner->executor = input_->executor();
  owner->leases.resize(ranges.size());
  owner->created.resize(ranges.size());
  owner->pending.reserve(ranges.size());
  owner->expected.reserve(ranges.size());

  gpu::DeviceReadBatch result;
  result.owner = owner;
  result.data.resize(ranges.size());
  std::vector<cudf::io::text::byte_range_info> bypass;
  std::vector<size_t> bypassIndices;
  bypass.reserve(ranges.size());
  bypassIndices.reserve(ranges.size());
  std::unordered_map<std::string, GpuEncodedCache::LeasePtr> reservations;
  try {
    for (size_t i = 0; i < ranges.size(); ++i) {
      const auto& range = ranges[i];
      VELOX_CHECK_GE(range.offset(), 0);
      VELOX_CHECK_GE(range.size(), 0);
      VELOX_CHECK_LE(range.offset(), size());
      VELOX_CHECK_LE(range.size(), size() - range.offset());
      if (range.size() == 0) {
        continue;
      }
      recordBytes("cudfGpuCacheEligibleBytes", range.size());
      const auto key = fmt::format(
          "parquet:{}:{}:{}",
          fileIdentity_->id(),
          range.offset(),
          range.size());
      constexpr uint64_t kDecoderWordBytes = sizeof(uint64_t);
      const auto padding =
          (kDecoderWordBytes - range.size() % kDecoderWordBytes) %
          kDecoderWordBytes;
      bool needsFill{false};
      if (auto it = reservations.find(key); it != reservations.end()) {
        owner->leases[i] = it->second;
      } else {
        owner->leases[i] = deviceCache_->reserve(
            key,
            range.size() + padding,
            stream.get(),
            fileIdentity_,
            needsFill);
        if (owner->leases[i]) {
          reservations.emplace(key, owner->leases[i]);
        }
      }
      owner->created[i] = needsFill;
      if (auto& lease = owner->leases[i]) {
        result.data[i] = gpu::DeviceByteView{
            static_cast<const uint8_t*>(lease->data()),
            static_cast<size_t>(range.size())};
        if (needsFill && padding) {
          CUDF_CUDA_TRY(cudaMemsetAsync(
              static_cast<uint8_t*>(lease->data()) + range.size(),
              0,
              padding,
              stream.get()));
        }
        recordBytes(
            needsFill ? "cudfGpuCacheFillBytes" : "cudfGpuCacheHitBytes",
            range.size());
      } else {
        bypass.emplace_back(range.offset(), range.size());
        bypassIndices.push_back(i);
        recordBytes("cudfGpuCacheBypassBytes", range.size());
      }
    }

    // Coalesce adjacent misses into host reads, then scatter directly into the
    // retained destinations. Submit every group before waiting for completion.
    static std::mutex submissionMutex;
    std::unique_lock submissionLock(submissionMutex, std::defer_lock);
    if (serializeSubmissions) {
      submissionLock.lock();
    }
    for (size_t i = 0; i < ranges.size();) {
      if (!owner->created[i]) {
        ++i;
        continue;
      }
      const auto offset = ranges[i].offset();
      size_t bytes = ranges[i].size();
      std::vector<std::pair<void*, size_t>> destinations{
          {owner->leases[i]->data(), static_cast<size_t>(ranges[i].size())}};
      size_t end = i + 1;
      // A single requested range remains indivisible; cap only aggregation of
      // adjacent ranges so large passes retain IO executor parallelism.
      constexpr size_t kMaxCoalescedBytes = 16L << 20;
      while (end < ranges.size() && owner->created[end] &&
             ranges[end].offset() == offset + bytes &&
             bytes <= kMaxCoalescedBytes &&
             ranges[end].size() <= kMaxCoalescedBytes - bytes) {
        destinations.emplace_back(
            owner->leases[end]->data(), ranges[end].size());
        bytes += ranges[end].size();
        ++end;
      }
      std::vector<GpuEncodedCache::LeasePtr> fillLeases(
          owner->leases.begin() + i, owner->leases.begin() + end);
      auto retainedBuffer = std::make_shared<std::unique_ptr<buffer>>();
      if (!owner->uploads) {
        owner->uploads = std::make_shared<HostUploadBuffers>(stream);
        owner->uploads->buffers.reserve(ranges.size());
      }
      owner->uploads->buffers.push_back(retainedBuffer);
      owner->expected.push_back(bytes);
      owner->pending.push_back(submitDeviceRead(
          input_->executor(),
          [source = owner->source,
           offset,
           bytes,
           destinations = std::move(destinations),
           stream,
           retainedBuffer,
           fillLeases = std::move(fillLeases)] {
            try {
              auto& buffer = *retainedBuffer;
              buffer = source->host_read(offset, bytes);
              VELOX_CHECK_EQ(buffer->size(), bytes, "Short cached input read");
              ensureCudaContextForThread();
              size_t consumed{0};
              for (const auto& [destination, length] : destinations) {
                CUDF_CUDA_TRY(cudaMemcpyAsync(
                    destination,
                    buffer->data() + consumed,
                    length,
                    cudaMemcpyDefault,
                    stream.get()));
                consumed += length;
              }
              source->recordBytes("cudfInputH2DBytes", bytes);
              // Publish on the producer, even if its consumer has not waited
              // yet. Other queries sharing a fill must not depend on that
              // consumer.
              for (const auto& lease : fillLeases) {
                lease->completeFill();
              }
              return bytes;
            } catch (...) {
              for (const auto& lease : fillLeases) {
                lease->failFill();
              }
              throw;
            }
          }));
      i = end;
    }
    if (!bypass.empty()) {
      owner->bypass = copyByteRangesAsync(
          shared_from_this(),
          {bypass.data(), bypass.size()},
          serializeSubmissions,
          stream,
          {outputResource, temporaryResource});
      for (size_t i = 0; i < bypassIndices.size(); ++i) {
        result.data[bypassIndices[i]] = owner->bypass.data[i];
      }
    }
    result.pending =
        std::async(std::launch::deferred, [owner] { owner->finish(); });
    result.ready = [owner](cuda::stream_ref consumer) {
      for (const auto& lease : owner->leases) {
        if (lease) {
          lease->waitForFillOn(consumer.get());
        }
      }
      owner->bypass.waitOn(consumer);
    };
    return result;
  } catch (...) {
    owner->error = std::current_exception();
    throw;
  }
}

bool BufferedInputDataSource::supports_device_read() const {
  return true;
}

void BufferedInputDataSource::readContiguous(
    size_t offset,
    size_t size,
    uint8_t* dst) {
  using namespace facebook::velox::dwio::common;
  // BufferedInput::read gives us a stream over the exact region.
  auto stream = input_->read(offset, size, LogType::FILE);
  VELOX_CHECK(stream != nullptr, "read() returned null stream");
  stream->readFully(reinterpret_cast<char*>(dst), size);
}

} // namespace facebook::velox::cudf_velox::connector::hive
