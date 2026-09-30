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
#include "velox/common/memory/MallocAllocator.h"

#include <folly/ScopeGuard.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace facebook::velox::memory {
namespace {

// Record backing operations while retaining libc storage and shared
// bookkeeping.
class RecordingAllocator : public MallocAllocator {
 public:
  explicit RecordingAllocator(const Options& options)
      : MallocAllocator(options) {}

  const std::vector<std::string>& operations() const {
    return operations_;
  }

 protected:
  void* rawAllocate(size_t bytes, uint16_t alignment) override {
    operations_.push_back("allocate");
    return MallocAllocator::rawAllocate(bytes, alignment);
  }

  void* rawAllocateZeroFilled(size_t bytes) override {
    operations_.push_back("zero");
    return MallocAllocator::rawAllocateZeroFilled(bytes);
  }

  void rawFree(void* pointer, size_t bytes) noexcept override {
    operations_.push_back("free");
    MallocAllocator::rawFree(pointer, bytes);
  }

  void* rawReallocate(
      void* pointer,
      size_t oldSize,
      size_t newSize,
      uint16_t alignment) override {
    operations_.push_back("reallocate");
    // Model a backing pool that requires allocate-copy-free for resizing.
    return nullptr;
  }

  void* dispatchAllocateContiguous(size_t maxBytes) override {
    operations_.push_back("contiguousAllocate");
    return MallocAllocator::dispatchAllocateContiguous(maxBytes);
  }

  void dispatchFreeContiguous(ContiguousAllocation& allocation) override {
    operations_.push_back("contiguousFree");
    MallocAllocator::dispatchFreeContiguous(allocation);
  }

 private:
  // Preserve call order to detect operations bypassing the backing hooks.
  std::vector<std::string> operations_;
};

TEST(MallocAllocatorTest, backingHooks) {
  MemoryAllocator::Options options;
  options.mallocContiguousEnabled = true;
  RecordingAllocator allocator{options};
  auto* data = static_cast<uint8_t*>(allocator.allocateBytes(64, 64));
  ASSERT_NE(data, nullptr);
  size_t dataBytes{64};
  SCOPE_EXIT {
    allocator.freeBytes(data, dataBytes);
  };
  EXPECT_EQ(reinterpret_cast<uintptr_t>(data) % 64, 0);
  std::memset(data, 0x5a, dataBytes);
  auto* resized =
      static_cast<uint8_t*>(allocator.reallocateBytes(data, 64, 128));
  ASSERT_NE(resized, nullptr);
  data = resized;
  dataBytes = 128;
  EXPECT_TRUE(
      std::all_of(data, data + 64, [](uint8_t byte) { return byte == 0x5a; }));
  EXPECT_EQ(allocator.totalUsedBytes(), dataBytes);

  auto* zeroed = static_cast<uint8_t*>(allocator.allocateZeroFilled(64));
  ASSERT_NE(zeroed, nullptr);
  EXPECT_TRUE(
      std::all_of(zeroed, zeroed + 64, [](uint8_t byte) { return byte == 0; }));
  allocator.freeBytes(zeroed, 64);

  ContiguousAllocation contiguous;
  SCOPE_EXIT {
    allocator.freeContiguous(contiguous);
  };
  ASSERT_TRUE(allocator.allocateContiguous(1, nullptr, contiguous));
  allocator.freeContiguous(contiguous);
  EXPECT_EQ(allocator.totalUsedBytes(), dataBytes);
  EXPECT_EQ(allocator.numAllocated(), 0);
  EXPECT_THAT(
      allocator.operations(),
      testing::ElementsAre(
          "allocate",
          "reallocate",
          "allocate",
          "free",
          "zero",
          "free",
          "contiguousAllocate",
          "contiguousFree"));
}

// Back allocations with a fixed arena to reproduce fragmentation independently
// of the system allocator's placement decisions.
class FragmentedAllocator : public MallocAllocator {
 public:
  static constexpr size_t kNumBlocks{8};

  FragmentedAllocator()
      : MallocAllocator(Options{}),
        blockBytes_(AllocationTraits::pageBytes(sizeClasses().back())),
        storage_(static_cast<uint8_t*>(std::malloc(kNumBlocks * blockBytes_))) {
    VELOX_CHECK_NOT_NULL(storage_);
  }

  ~FragmentedAllocator() override {
    EXPECT_EQ(liveBlocks(), 0);
    std::free(storage_);
  }

  size_t allocationAttempts() const {
    return allocationAttempts_;
  }

  size_t freedBlocks() const {
    return freedBlocks_;
  }

  size_t liveBlocks() const {
    return std::count(occupied_.begin(), occupied_.end(), true);
  }

 protected:
  // Preserve size-class boundaries so requests can use disjoint free blocks.
  bool coalesceNonContiguousRuns() const override {
    return false;
  }

  void* rawAllocate(size_t bytes, uint16_t alignment) override {
    ++allocationAttempts_;
    VELOX_CHECK_LE(alignment, kMinAlignment);
    const auto blocks = bits::roundUp(bytes, blockBytes_) / blockBytes_;
    for (size_t start = 0; start + blocks <= kNumBlocks; ++start) {
      if (std::none_of(
              occupied_.begin() + start,
              occupied_.begin() + start + blocks,
              [](bool occupied) { return occupied; })) {
        std::fill_n(occupied_.begin() + start, blocks, true);
        return storage_ + start * blockBytes_;
      }
    }
    return nullptr;
  }

  void rawFree(void* pointer, size_t bytes) noexcept override {
    const auto start =
        (static_cast<uint8_t*>(pointer) - storage_) / blockBytes_;
    const auto blocks = bits::roundUp(bytes, blockBytes_) / blockBytes_;
    EXPECT_LE(start + blocks, kNumBlocks);
    for (size_t i = start; i < start + blocks; ++i) {
      EXPECT_TRUE(occupied_[i]);
      occupied_[i] = false;
      ++freedBlocks_;
    }
  }

 private:
  // Match the largest noncontiguous size class for each arena block.
  const size_t blockBytes_;
  // Own a contiguous arena whose individual blocks can remain occupied.
  uint8_t* const storage_;
  // Track backing ownership separately from MallocAllocator's accounting.
  std::array<bool, kNumBlocks> occupied_{};
  // Count failed requests as well as successful backing allocations.
  size_t allocationAttempts_{0};
  // Count blocks returned by normal cleanup and partial-allocation rollback.
  size_t freedBlocks_{0};
};

// Leave alternating occupied blocks to prevent coalesced backing requests.
testing::AssertionResult fragmentBacking(
    FragmentedAllocator& allocator,
    std::array<Allocation, FragmentedAllocator::kNumBlocks>& allocations) {
  for (auto& allocation : allocations) {
    if (!allocator.allocateNonContiguous(
            allocator.sizeClasses().back(), allocation)) {
      return testing::AssertionFailure() << "Could not fill the backing arena";
    }
    std::memset(allocation.runAt(0).data(), 0x5a, allocation.byteSize());
  }
  for (size_t i = 0; i < allocations.size(); i += 2) {
    allocator.freeNonContiguous(allocations[i]);
  }
  return testing::AssertionSuccess();
}

TEST(MallocAllocatorTest, allocateFromFragmentedBacking) {
  FragmentedAllocator allocator;
  std::array<Allocation, FragmentedAllocator::kNumBlocks> allocations;
  Allocation result;
  SCOPE_EXIT {
    allocator.freeNonContiguous(result);
    for (auto& allocation : allocations) {
      allocator.freeNonContiguous(allocation);
    }
  };
  const auto blockPages = allocator.sizeClasses().back();
  ASSERT_TRUE(fragmentBacking(allocator, allocations));
  ASSERT_EQ(allocator.liveBlocks(), 4);
  // Four isolated free blocks can satisfy the logical request together.
  ASSERT_TRUE(allocator.allocateNonContiguous(4 * blockPages, result));
  EXPECT_EQ(result.numPages(), 4 * blockPages);
  EXPECT_EQ(result.numRuns(), 4);
  for (int i = 0; i < result.numRuns(); ++i) {
    EXPECT_EQ(result.runAt(i).numPages(), blockPages);
    std::memset(result.runAt(i).data(), 0xa5, result.runAt(i).numBytes());
  }
  for (size_t i = 1; i < allocations.size(); i += 2) {
    const auto* data = allocations[i].runAt(0).data<uint8_t>();
    EXPECT_TRUE(
        std::all_of(data, data + allocations[i].byteSize(), [](uint8_t byte) {
          return byte == 0x5a;
        }));
  }
  EXPECT_EQ(allocator.numAllocated(), 8 * blockPages);
  EXPECT_EQ(
      allocator.totalUsedBytes(), 8 * AllocationTraits::pageBytes(blockPages));
}

TEST(MallocAllocatorTest, fragmentedBackingFailureRollsBack) {
  FragmentedAllocator allocator;
  std::array<Allocation, FragmentedAllocator::kNumBlocks> allocations;
  Allocation result;
  SCOPE_EXIT {
    allocator.freeNonContiguous(result);
    for (auto& allocation : allocations) {
      allocator.freeNonContiguous(allocation);
    }
  };
  const auto blockPages = allocator.sizeClasses().back();
  const auto blockBytes = AllocationTraits::pageBytes(blockPages);
  ASSERT_TRUE(fragmentBacking(allocator, allocations));
  // Exhaust the four free blocks before the fifth backing allocation fails.
  // Repeating the failure must preserve both ownership and capacity
  // accounting.
  for (int i = 0; i < 2; ++i) {
    const auto attemptsBefore = allocator.allocationAttempts();
    const auto freedBefore = allocator.freedBlocks();
    EXPECT_FALSE(allocator.allocateNonContiguous(5 * blockPages, result));
    EXPECT_EQ(allocator.allocationAttempts() - attemptsBefore, 5);
    EXPECT_EQ(allocator.freedBlocks() - freedBefore, 4);
    EXPECT_EQ(result.numRuns(), 0);
    EXPECT_EQ(result.numPages(), 0);
    EXPECT_EQ(allocator.liveBlocks(), 4);
    EXPECT_EQ(allocator.numAllocated(), 4 * blockPages);
    EXPECT_EQ(allocator.totalUsedBytes(), 4 * blockBytes);
  }
  ASSERT_TRUE(allocator.allocateNonContiguous(4 * blockPages, result));
  EXPECT_EQ(result.numRuns(), 4);
  EXPECT_EQ(allocator.freeNonContiguous(result), 4 * blockBytes);
  for (auto& allocation : allocations) {
    allocator.freeNonContiguous(allocation);
  }
  EXPECT_EQ(allocator.liveBlocks(), 0);
  EXPECT_EQ(allocator.numAllocated(), 0);
  EXPECT_EQ(allocator.totalUsedBytes(), 0);
}

// Back size-class runs with 64 KiB blocks and opt into subdividing failed runs,
// modeling a bounded pool whose free space is scattered below 1 MiB.
class SplittingAllocator : public MallocAllocator {
 public:
  static constexpr size_t kNumBlocks{32};
  static constexpr MachinePageCount kBlockPages{16};

  SplittingAllocator()
      : MallocAllocator(Options{}),
        storage_(
            static_cast<uint8_t*>(std::malloc(
                kNumBlocks * AllocationTraits::pageBytes(kBlockPages)))) {
    VELOX_CHECK_NOT_NULL(storage_);
  }

  ~SplittingAllocator() override {
    EXPECT_EQ(liveBlocks(), 0);
    std::free(storage_);
  }

  size_t allocationAttempts() const {
    return allocationAttempts_;
  }

  size_t freedBlocks() const {
    return freedBlocks_;
  }

  size_t liveBlocks() const {
    return std::count(occupied_.begin(), occupied_.end(), true);
  }

  // Marks every block occupied except those at 'freeIndices', leaving
  // fragmented free space without going through the allocator.
  void occupyAllBut(const std::vector<size_t>& freeIndices) {
    std::fill(occupied_.begin(), occupied_.end(), true);
    for (auto i : freeIndices) {
      occupied_[i] = false;
    }
  }

  // Returns the blocks occupied by occupyAllBut() before destruction.
  void releaseAll() {
    std::fill(occupied_.begin(), occupied_.end(), false);
  }

 protected:
  bool coalesceNonContiguousRuns() const override {
    return false;
  }

  bool splitNonContiguousRunsOnFailure() const override {
    return true;
  }

  // First fit over whole 64 KiB blocks; a run larger than one block needs
  // adjacent free blocks.
  void* rawAllocate(size_t bytes, uint16_t alignment) override {
    ++allocationAttempts_;
    VELOX_CHECK_LE(alignment, kMinAlignment);
    const auto blockBytes = AllocationTraits::pageBytes(kBlockPages);
    const auto blocks = bits::roundUp(bytes, blockBytes) / blockBytes;
    for (size_t start = 0; start + blocks <= kNumBlocks; ++start) {
      if (std::none_of(
              occupied_.begin() + start,
              occupied_.begin() + start + blocks,
              [](bool occupied) { return occupied; })) {
        std::fill_n(occupied_.begin() + start, blocks, true);
        return storage_ + start * blockBytes;
      }
    }
    return nullptr;
  }

  void rawFree(void* pointer, size_t bytes) noexcept override {
    const auto blockBytes = AllocationTraits::pageBytes(kBlockPages);
    const auto start = (static_cast<uint8_t*>(pointer) - storage_) / blockBytes;
    const auto blocks = bits::roundUp(bytes, blockBytes) / blockBytes;
    EXPECT_LE(start + blocks, kNumBlocks);
    for (size_t i = start; i < start + blocks; ++i) {
      EXPECT_TRUE(occupied_[i]);
      occupied_[i] = false;
      ++freedBlocks_;
    }
  }

 private:
  // Own a contiguous arena of 64 KiB blocks.
  uint8_t* const storage_;
  // Track backing ownership separately from MallocAllocator's accounting.
  std::array<bool, kNumBlocks> occupied_{};
  // Count failed requests as well as successful backing allocations.
  size_t allocationAttempts_{0};
  // Count blocks returned by normal cleanup and partial-allocation rollback.
  size_t freedBlocks_{0};
};

// Every other block free: 1 MiB is free in total, but no two free blocks are
// adjacent, so no run larger than 64 KiB fits.
std::vector<size_t> alternateBlocks() {
  std::vector<size_t> indices;
  for (size_t i = 0; i < SplittingAllocator::kNumBlocks; i += 2) {
    indices.push_back(i);
  }
  return indices;
}

TEST(MallocAllocatorTest, splitFailedRunIntoFragmentedBacking) {
  SplittingAllocator allocator;
  allocator.occupyAllBut(alternateBlocks());
  SCOPE_EXIT {
    allocator.releaseAll();
  };
  Allocation allocation;
  // A 1 MiB size-class run fails; halving it down to 64 KiB uses the holes.
  ASSERT_TRUE(allocator.allocateNonContiguous(256, allocation));
  EXPECT_EQ(allocation.numPages(), 256);
  EXPECT_EQ(allocation.numRuns(), 16);
  for (int i = 0; i < allocation.numRuns(); ++i) {
    EXPECT_EQ(allocation.runAt(i).numPages(), SplittingAllocator::kBlockPages);
    std::memset(
        allocation.runAt(i).data(), 0x5a, allocation.runAt(i).numBytes());
  }
  EXPECT_EQ(allocator.numAllocated(), 256);
  EXPECT_EQ(allocator.totalUsedBytes(), AllocationTraits::pageBytes(256));
  allocator.freeNonContiguous(allocation);
  EXPECT_EQ(allocator.numAllocated(), 0);
  EXPECT_EQ(allocator.totalUsedBytes(), 0);
  EXPECT_EQ(allocator.liveBlocks(), SplittingAllocator::kNumBlocks / 2);
}

TEST(MallocAllocatorTest, splitRunsRespectMinimumSizeClass) {
  SplittingAllocator allocator;
  allocator.occupyAllBut(alternateBlocks());
  SCOPE_EXIT {
    allocator.releaseAll();
  };
  Allocation allocation;
  // Only 64 KiB runs exist, so a caller requiring 128 KiB runs must fail
  // without leaking backing or capacity.
  EXPECT_FALSE(allocator.allocateNonContiguous(256, allocation, nullptr, 32));
  EXPECT_TRUE(allocation.empty());
  EXPECT_EQ(allocator.numAllocated(), 0);
  EXPECT_EQ(allocator.liveBlocks(), SplittingAllocator::kNumBlocks / 2);
  ASSERT_TRUE(allocator.allocateNonContiguous(256, allocation, nullptr, 16));
  EXPECT_EQ(allocation.numPages(), 256);
  for (int i = 0; i < allocation.numRuns(); ++i) {
    EXPECT_EQ(allocation.runAt(i).numPages(), SplittingAllocator::kBlockPages);
  }
  allocator.freeNonContiguous(allocation);
  EXPECT_EQ(allocator.numAllocated(), 0);
}

TEST(MallocAllocatorTest, partialSplitFailureRollsBack) {
  SplittingAllocator allocator;
  // Eight isolated 64 KiB holes: half of what a 1 MiB request needs.
  std::vector<size_t> holes;
  for (size_t i = 0; i < 16; i += 2) {
    holes.push_back(i);
  }
  allocator.occupyAllBut(holes);
  SCOPE_EXIT {
    allocator.releaseAll();
  };
  Allocation allocation;
  const auto freedBefore = allocator.freedBlocks();
  EXPECT_FALSE(allocator.allocateNonContiguous(256, allocation));
  // Subdivision found all eight holes before running out, then returned them.
  EXPECT_EQ(allocator.freedBlocks() - freedBefore, holes.size());
  EXPECT_TRUE(allocation.empty());
  EXPECT_EQ(allocator.numAllocated(), 0);
  EXPECT_EQ(allocator.totalUsedBytes(), 0);
  EXPECT_EQ(
      allocator.liveBlocks(), SplittingAllocator::kNumBlocks - holes.size());
}

TEST(MallocAllocatorTest, defaultCoalescesSizeClassRuns) {
  MallocAllocator allocator{MemoryAllocator::Options{}};
  Allocation allocation;
  SCOPE_EXIT {
    allocator.freeNonContiguous(allocation);
  };
  const auto blockPages = allocator.sizeClasses().back();
  ASSERT_TRUE(allocator.allocateNonContiguous(4 * blockPages, allocation));
  ASSERT_EQ(allocation.numRuns(), 1);
  EXPECT_EQ(allocation.runAt(0).numPages(), 4 * blockPages);
  EXPECT_EQ(allocator.numAllocated(), 4 * blockPages);
  EXPECT_EQ(allocator.totalUsedBytes(), allocation.byteSize());

  // Replacing an allocation must return its old backing and account only for
  // the new size, including requests with more than one size class.
  ASSERT_TRUE(allocator.allocateNonContiguous(2 * blockPages + 1, allocation));
  ASSERT_EQ(allocation.numRuns(), 2);
  EXPECT_EQ(allocation.runAt(0).numPages(), 2 * blockPages);
  EXPECT_EQ(allocation.runAt(1).numPages(), 1);
  EXPECT_EQ(allocator.numAllocated(), 2 * blockPages + 1);
  EXPECT_EQ(allocator.totalUsedBytes(), allocation.byteSize());
  EXPECT_EQ(
      allocator.freeNonContiguous(allocation),
      AllocationTraits::pageBytes(2 * blockPages + 1));
  EXPECT_EQ(allocator.numAllocated(), 0);
  EXPECT_EQ(allocator.totalUsedBytes(), 0);
}

TEST(MallocAllocatorTest, separateRunsHonorSizeClassesAndReplacement) {
  FragmentedAllocator allocator;
  Allocation allocation;
  SCOPE_EXIT {
    allocator.freeNonContiguous(allocation);
  };
  const auto blockPages = allocator.sizeClasses().back();
  for (const auto minSizeClass : {1, 16}) {
    ASSERT_TRUE(allocator.allocateNonContiguous(
        2 * blockPages + 1, allocation, nullptr, minSizeClass));
    ASSERT_EQ(allocation.numRuns(), 3);
    std::vector<MachinePageCount> runPages;
    for (int i = 0; i < allocation.numRuns(); ++i) {
      runPages.push_back(allocation.runAt(i).numPages());
    }
    EXPECT_THAT(
        runPages, testing::ElementsAre(blockPages, blockPages, minSizeClass));
    EXPECT_EQ(allocator.liveBlocks(), 3);
    EXPECT_EQ(allocator.numAllocated(), 2 * blockPages + minSizeClass);
    EXPECT_EQ(allocator.totalUsedBytes(), allocation.byteSize());
  }
  ASSERT_TRUE(allocator.allocateNonContiguous(0, allocation));
  EXPECT_EQ(allocation.numRuns(), 0);
  EXPECT_EQ(allocator.liveBlocks(), 0);
  EXPECT_EQ(allocator.numAllocated(), 0);
  EXPECT_EQ(allocator.totalUsedBytes(), 0);
}

} // namespace
} // namespace facebook::velox::memory
