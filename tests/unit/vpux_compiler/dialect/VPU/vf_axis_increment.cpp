//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/vf_axis_increment.hpp"

#include "common/utils.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace vpux;
using namespace VPU;

using MLIR_VPU_VFAxisIncrement = MLIR_UnitBase;

TEST_F(MLIR_VPU_VFAxisIncrement, VF_SpatialDim) {
    auto axisIncrement = VPU::getVFAxisIncrement(Dims4D::Act::H);

    EXPECT_TRUE(axisIncrement != nullptr);

    EXPECT_EQ(axisIncrement->getMiddleValue(2, 10), 6);

    int64_t value = 10;
    axisIncrement->increasedValue(value, 16);
    axisIncrement->increasedValue(value, 16);
    EXPECT_EQ(value, 12);
    axisIncrement->decreasedValue(value, 8);
    EXPECT_EQ(value, 11);

    EXPECT_EQ(axisIncrement->getLimitValue({4, 6, 2}, {}), 2);
    EXPECT_EQ(axisIncrement->getLimitValue({}, {4, 6, 2}), 2);
    EXPECT_EQ(axisIncrement->getLimitValue({474, 474, 474}, {2048}), 474);
    EXPECT_EQ(axisIncrement->getLimitValue({474, 474, 474}, {4}), 4);
}

TEST_F(MLIR_VPU_VFAxisIncrement, VF_ChannelDim) {
    auto axisIncrement = VPU::getVFAxisIncrement(Dims4D::Act::C);

    EXPECT_TRUE(axisIncrement != nullptr);

    EXPECT_EQ(axisIncrement->getMiddleValue(2, 16), 8);

    int64_t value = 2;
    axisIncrement->increasedValue(value, 8);
    EXPECT_EQ(value, 4);
    value = 16;
    axisIncrement->decreasedValue(value, 2);
    EXPECT_EQ(value, 8);

    EXPECT_EQ(axisIncrement->getLimitValue({12, 8, 32}, {}), 4);
    EXPECT_EQ(axisIncrement->getLimitValue({}, {12, 8, 32}), 8);
    EXPECT_EQ(axisIncrement->getLimitValue({474, 474, 474}, {2048}), 474);
    EXPECT_EQ(axisIncrement->getLimitValue({474, 474, 474}, {4}), 2);
}
