//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>

#include "vpux/compiler/dialect/VPUIP/utils/cache_utils.hpp"

#include "vpux/utils/core/range.hpp"

using namespace vpux;

TEST(MLIR_CacheUtils, ShaveL2CacheSimulator) {
    const size_t CAPACITY = 1024;

    VPUIP::ShaveL2CacheSimulator cache(CAPACITY);
    EXPECT_EQ(cache.getCapacity(), CAPACITY);

    std::unordered_map<std::string, size_t> KERNELS = {{"K768", 768}, {"K512", 512},   {"K256", 256},
                                                       {"K128", 128}, {"K128_2", 128}, {"K257", 257}};
    const auto load = [&](const std::string& kernel) {
        cache.loadKernel(kernel, KERNELS[kernel]);
        EXPECT_EQ(cache.isLoaded(kernel), true);
    };
    const auto leastRecent = [&]() -> std::string {
        return cache.getUsageHistory().back();
    };
    const auto checkOrder = [&](const std::vector<std::string>& expectedOrder) {
        for (const auto& [id, kernel] : cache.getUsageHistory() | indexed) {
            EXPECT_EQ(kernel, expectedOrder[id]);
        }
    };

    load("K512");

    load("K256");
    EXPECT_EQ(leastRecent(), "K512");

    // After this lookup K512 is more recent
    load("K512");
    EXPECT_EQ(leastRecent(), "K256");

    load("K128");
    checkOrder({"K128", "K512", "K256"});

    // K768 is too big to fit cache now, so evict least recent tasks
    load("K768");
    checkOrder({"K768", "K128"});

    load("K128");
    EXPECT_EQ(leastRecent(), "K768");

    load("K257");
    checkOrder({"K257", "K128"});

    load("K128_2");
    checkOrder({"K128_2", "K257", "K128"});

    load("K256");
    checkOrder({"K256", "K128_2", "K257", "K128"});

    EXPECT_ANY_THROW(cache.loadKernel("K1025", 1025));
}
