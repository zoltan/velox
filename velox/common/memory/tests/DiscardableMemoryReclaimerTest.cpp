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

#include "velox/common/memory/DiscardableMemoryReclaimer.h"
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/memory/Memory.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/common/memory/tests/SharedArbitratorTestUtil.h"

#include <gtest/gtest.h>
#include <atomic>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <vector>

namespace facebook::velox::memory {
namespace {
constexpr uint64_t kMegabyte = 1ULL << 20;
constexpr uint64_t kCapacity = 256 * kMegabyte;
constexpr uint64_t kBlock = 4 * kMegabyte;

// Holds actual pool allocations and records reclamation independently of
// policy.
struct AllocationState {
  MemoryPool* pool{nullptr};
  mutable std::mutex mutex;
  std::vector<void*> buffers;
  bool pinned{false};
  bool allowReclaim{true};
  std::function<void()> afterReclaim;
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> aborts{0};
  std::atomic<uint64_t> lastTarget{0};

  uint64_t eligible() const {
    std::lock_guard lock(mutex);
    return pinned ? 0 : buffers.size() * kBlock;
  }

  void allocate(uint64_t bytes) {
    for (uint64_t allocated = 0; allocated < bytes; allocated += kBlock) {
      auto* buffer = pool->allocate(kBlock);
      std::lock_guard lock(mutex);
      buffers.push_back(buffer);
    }
  }

  uint64_t free(uint64_t target, bool all = false) {
    std::lock_guard lock(mutex);
    if (!all && (pinned || !allowReclaim)) {
      return 0;
    }
    uint64_t freed{0};
    while (!buffers.empty() && (all || freed < target)) {
      pool->free(buffers.back(), kBlock);
      buffers.pop_back();
      freed += kBlock;
    }
    return freed;
  }
};

// Uses the same reclamation behavior for operator and discardable
// participants.
template <typename Base>
class RecordingReclaimer final : public Base {
 public:
  explicit RecordingReclaimer(std::shared_ptr<AllocationState> state)
      : Base(0), state_(std::move(state)) {}

  bool reclaimableBytes(const MemoryPool&, uint64_t& bytes) const override {
    bytes = state_->eligible();
    return true;
  }

  uint64_t reclaim(
      MemoryPool*,
      uint64_t target,
      uint64_t,
      MemoryReclaimer::Stats& stats) override {
    ++state_->calls;
    state_->lastTarget = target;
    const auto freed = state_->free(target);
    stats.reclaimedBytes += freed;
    if (state_->afterReclaim) {
      state_->afterReclaim();
    }
    return freed;
  }

  void abort(MemoryPool*, const std::exception_ptr&) override {
    ++state_->aborts;
    state_->free(0, true);
  }

 private:
  const std::shared_ptr<AllocationState> state_;
};

// Keeps allocations alive through assertions and releases them before pools.
struct Participant {
  std::shared_ptr<AllocationState> state;
  std::shared_ptr<MemoryPool> root;
  std::shared_ptr<MemoryPool> leaf;

  ~Participant() {
    state->free(0, true);
  }
};

class DiscardableMemoryReclaimerTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    SharedArbitrator::registerFactory();
  }

  static void TearDownTestSuite() {
    SharedArbitrator::unregisterFactory();
  }

  void setup(std::optional<bool> enabled = true, bool withoutSpill = false) {
    MemoryManager::Options options;
    options.allocatorCapacity = kCapacity;
    options.arbitratorKind = "SHARED";
    options.extraArbitratorConfigs = {
        {"memory-pool-initial-capacity", "0B"},
        {"memory-pool-reserved-capacity", "0B"},
        {"fast-exponential-growth-capacity-limit", "0B"},
        {"slow-capacity-grow-pct", "0"},
        {"memory-pool-min-free-capacity", "0B"},
        {"memory-pool-min-free-capacity-pct", "0"},
        {"memory-pool-min-reclaim-bytes", "0B"},
        {"memory-pool-min-reclaim-pct", "0"},
        {"global-arbitration-enabled", "true"},
        {"global-arbitration-memory-reclaim-pct", "50"},
        {"global-arbitration-without-spill", withoutSpill ? "true" : "false"},
        {"max-memory-arbitration-time", "5s"}};
    if (enabled.has_value()) {
      options.extraArbitratorConfigs[std::string(
          SharedArbitrator::ExtraConfig::kBoundedDiscardableReclaim)] =
          enabled.value() ? "true" : "false";
    }
    manager_ = std::make_unique<MemoryManager>(options);
  }

  std::unique_ptr<Participant> participant(
      const std::string& name,
      bool discardable,
      uint64_t capacity = kCapacity) {
    auto result = std::make_unique<Participant>();
    result->state = std::make_shared<AllocationState>();
    std::unique_ptr<MemoryReclaimer> reclaimer;
    if (discardable) {
      reclaimer =
          std::make_unique<RecordingReclaimer<DiscardableMemoryReclaimer>>(
              result->state);
    } else {
      reclaimer =
          std::make_unique<RecordingReclaimer<MemoryReclaimer>>(result->state);
    }
    result->root = manager_->addRootPool(name, capacity, std::move(reclaimer));
    result->leaf = result->root->addLeafChild("data");
    result->state->pool = result->leaf.get();
    return result;
  }

  std::unique_ptr<MemoryManager> manager_;
};

TEST_F(DiscardableMemoryReclaimerTest, prefersCacheBeforeSpillOrAbort) {
  for (bool withoutSpill : {false, true}) {
    SCOPED_TRACE(withoutSpill);
    setup(true, withoutSpill);
    auto cache = participant("cache", true);
    auto query = participant("query", false);
    auto requester = participant("requester", false);
    cache->state->allocate(64 * kMegabyte);
    query->state->allocate(192 * kMegabyte);
    requester->state->allocate(8 * kMegabyte);
    EXPECT_GT(cache->state->calls, 0);
    EXPECT_LE(cache->state->lastTarget, 24 * kMegabyte);
    EXPECT_GE(cache->root->usedBytes(), 40 * kMegabyte);
    EXPECT_EQ(query->state->calls, 0);
    EXPECT_EQ(query->state->aborts, 0);
    EXPECT_EQ(requester->root->usedBytes(), 8 * kMegabyte);
  }
}

TEST_F(DiscardableMemoryReclaimerTest, admissionDoesNotReclaimQueryState) {
  setup();
  auto query = participant("query", false);
  auto cache = participant("cache", true);
  query->state->allocate(kCapacity);
  VELOX_ASSERT_THROW(
      cache->state->allocate(kBlock), "disposable cache admission");
  EXPECT_EQ(query->state->calls, 0);
  EXPECT_EQ(query->state->aborts, 0);
  EXPECT_EQ(query->root->usedBytes(), kCapacity);
}

TEST_F(DiscardableMemoryReclaimerTest, fullCacheParticipantRefusesAdmission) {
  setup();
  auto cache = participant("cache", true, 16 * kMegabyte);
  cache->state->allocate(16 * kMegabyte);
  VELOX_ASSERT_THROW(
      cache->state->allocate(kBlock), "disposable cache admission");
  EXPECT_EQ(cache->state->calls, 0);
  EXPECT_EQ(cache->root->usedBytes(), 16 * kMegabyte);
}

TEST_F(DiscardableMemoryReclaimerTest, admissionUsesUnusedReservations) {
  setup();
  auto query = participant("query", false);
  auto cache = participant("cache", true);
  query->state->allocate(kCapacity);
  query->state->free(kCapacity);
  cache->state->allocate(32 * kMegabyte);
  EXPECT_EQ(cache->root->usedBytes(), 32 * kMegabyte);
  EXPECT_EQ(query->state->calls, 0);
  EXPECT_EQ(query->state->aborts, 0);
}

TEST_F(DiscardableMemoryReclaimerTest, pinnedCacheFallsBackToOperatorReclaim) {
  setup();
  auto cache = participant("cache", true);
  auto query = participant("query", false);
  auto requester = participant("requester", false);
  cache->state->allocate(64 * kMegabyte);
  cache->state->pinned = true;
  query->state->allocate(192 * kMegabyte);
  requester->state->allocate(8 * kMegabyte);
  EXPECT_EQ(cache->root->usedBytes(), 64 * kMegabyte);
  EXPECT_EQ(cache->state->calls, 0);
  EXPECT_GT(query->state->calls, 0);
}

TEST_F(DiscardableMemoryReclaimerTest, nonProgressingCacheFallsBack) {
  setup();
  auto cache = participant("cache", true);
  auto query = participant("query", false);
  auto requester = participant("requester", false);
  cache->state->allocate(64 * kMegabyte);
  // Model bytes becoming unavailable between candidate sampling and reclaim.
  cache->state->allowReclaim = false;
  query->state->allocate(192 * kMegabyte);
  requester->state->allocate(8 * kMegabyte);
  EXPECT_EQ(cache->root->usedBytes(), 64 * kMegabyte);
  EXPECT_GT(cache->state->calls, 0);
  EXPECT_EQ(cache->state->aborts, 0);
  EXPECT_GT(query->state->calls, 0);
}

TEST_F(DiscardableMemoryReclaimerTest, concurrentCapacityReturnIsProgress) {
  setup();
  auto cache = participant("cache", true);
  auto query = participant("query", false);
  auto requester = participant("requester", false);
  cache->state->allocate(64 * kMegabyte);
  query->state->allocate(192 * kMegabyte);
  // Model another allocator returning the freed reservation before the
  // reclaiming participant can shrink it itself.
  cache->state->afterReclaim = [&] {
    manager_->arbitrator()->shrinkCapacity(0, false, false);
  };
  requester->state->allocate(8 * kMegabyte);
  // Let the global controller finish before inspecting its fallback decision.
  test::SharedArbitratorTestHelper(
      static_cast<SharedArbitrator*>(manager_->arbitrator()))
      .waitForGlobalArbitrationToFinish();
  EXPECT_EQ(query->state->calls, 0);
  EXPECT_EQ(query->state->aborts, 0);
}

TEST_F(DiscardableMemoryReclaimerTest, disabledPolicyRetainsStockSelection) {
  for (const auto enabled :
       {std::optional<bool>{false}, std::optional<bool>{}}) {
    SCOPED_TRACE(enabled.has_value());
    setup(enabled);
    auto cache = participant("cache", true);
    auto query = participant("query", false);
    auto requester = participant("requester", false);
    cache->state->allocate(64 * kMegabyte);
    query->state->allocate(192 * kMegabyte);
    requester->state->allocate(8 * kMegabyte);
    EXPECT_GT(query->state->calls, 0);
  }
}

TEST_F(DiscardableMemoryReclaimerTest, concurrentRequestsShareCacheCapacity) {
  setup();
  auto cache = participant("cache", true);
  auto query = participant("query", false);
  cache->state->allocate(128 * kMegabyte);
  query->state->allocate(128 * kMegabyte);
  std::vector<std::unique_ptr<Participant>> requesters;
  for (int i = 0; i < 8; ++i) {
    requesters.push_back(participant("requester" + std::to_string(i), false));
  }
  std::promise<void> ready;
  auto start = ready.get_future().share();
  std::vector<std::future<void>> pending;
  for (const auto& requester : requesters) {
    pending.push_back(
        std::async(std::launch::async, [start, state = requester->state] {
          start.wait();
          state->allocate(8 * kMegabyte);
        }));
  }
  ready.set_value();
  for (auto& work : pending) {
    work.get();
  }
  test::SharedArbitratorTestHelper(
      static_cast<SharedArbitrator*>(manager_->arbitrator()))
      .waitForGlobalArbitrationToFinish();
  for (const auto& requester : requesters) {
    EXPECT_EQ(requester->root->usedBytes(), 8 * kMegabyte);
  }
  EXPECT_EQ(query->state->calls, 0);
  EXPECT_EQ(query->state->aborts, 0);
}

} // namespace
} // namespace facebook::velox::memory
