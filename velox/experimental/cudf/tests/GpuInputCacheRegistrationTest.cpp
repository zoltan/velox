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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfMemoryArbitration.h"
#include "velox/experimental/cudf/exec/EncodedInputGpuCache.h"
#include "velox/experimental/cudf/exec/ToCudf.h"

#include "velox/common/caching/AsyncDataCache.h"
#include "velox/common/caching/SsdCache.h"
#include "velox/common/memory/CustomMemoryResourceRegistry.h"
#include "velox/common/memory/Memory.h"
#include "velox/core/QueryCtx.h"

#include <gtest/gtest.h>

namespace facebook::velox::cudf_velox {
TEST(GpuInputCacheRegistrationTest, registrationUsesAccountedBackend) {
  memory::MemoryManager::initialize({});
  auto hostCache =
      cache::AsyncDataCache::create(memory::memoryManager()->allocator());
  cache::AsyncDataCache::setInstance(hostCache.get());
  auto resource = createCudfCustomMemoryResource(128L << 20, true);
  memory::CustomMemoryResourceRegistry::global().insert(
      std::string{kCudfMemoryResourceTag}, resource);
  registerCudfMemoryResource(128L << 20);
  CudfConfig::getInstance().initialize(
      {{CudfConfig::kGpuInputCacheEnabled, "false"}});
  registerCudf();
  EXPECT_FALSE(encodedInputGpuCache());
  unregisterCudf();
  CudfConfig::getInstance().initialize(
      {{CudfConfig::kGpuInputCacheEnabled, "true"}});
  cache::AsyncDataCache::setInstance(nullptr);
  registerCudf();
  EXPECT_FALSE(encodedInputGpuCache());
  unregisterCudf();
  cache::AsyncDataCache::setInstance(hostCache.get());
  registerCudf();
  EXPECT_TRUE(encodedInputGpuCache());
  {
    auto query =
        core::QueryCtx::Builder().asyncDataCache(hostCache.get()).build();
    attachCudfMemoryPool(*query);
    EXPECT_EQ(
        query->customPool(std::string{kCudfMemoryResourceTag})->arbitrator(),
        resource->arbitrator());
  }
  unregisterCudf();
  EXPECT_FALSE(encodedInputGpuCache());
  cache::AsyncDataCache::setInstance(nullptr);
  hostCache->shutdown();
}
} // namespace facebook::velox::cudf_velox
