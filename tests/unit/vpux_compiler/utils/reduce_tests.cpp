//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <gtest/gtest.h>
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"

using namespace vpux;

TEST(MLIR_ReduceTest, calculateReducedOutputLayout) {
    // No alignment single axis tiling
    {
        mlir::SmallVector<std::tuple<DimsOrder, mlir::SmallVector<int64_t>, DimsOrder>> dimOrderVec = {
                {/*inputDimOrder*/ DimsOrder::fromCode(0x1234), /*axes*/ {1, 2},
                 /*outputDimOrder*/ DimsOrder::fromCode(0x12)},
                {/*inputDimOrder*/ DimsOrder::fromCode(0x1432), /*axes*/ {1, 2},
                 /*outputDimOrder*/ DimsOrder::fromCode(0x21)},
                {/*inputDimOrder*/ DimsOrder::fromCode(0x1324), /*axes*/ {2},
                 /*outputDimOrder*/ DimsOrder::fromCode(0x123)},
                {/*inputDimOrder*/ DimsOrder::fromCode(0x13452), /*axes*/ {2, 3},
                 /*outputDimOrder*/ DimsOrder::fromCode(0x123)},
                {/*inputDimOrder*/ DimsOrder::fromCode(0x13452), /*axes*/ {1, 5},
                 /*outputDimOrder*/ DimsOrder::fromCode(0x231)},
                {/*inputDimOrder*/ DimsOrder::fromCode(0x12345), /*axes*/ {1, 2, 3, 4},
                 /*outputDimOrder*/ DimsOrder::fromCode(0x1)}};

        for (auto it : dimOrderVec) {
            auto inputDimOrder = std::get<0>(it);
            auto axes = std::get<1>(it);
            auto actualOutputDimOrder = vpux::IE::calculateReducedOutputLayout(inputDimOrder, axes);
            EXPECT_EQ(actualOutputDimOrder, std::get<2>(it));
        }
    }
}
