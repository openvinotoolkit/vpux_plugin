//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
#include <gtest/gtest.h>
#include <climits>
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;

namespace {
struct InstructionListTableStruct {
    SmallVector<int32_t> range;
    SmallVector<int32_t> shift;
    SmallVector<int32_t> bias;
    SmallVector<int32_t> expectedOutput;
};
class MLIR_InstructionListTableTest : public testing::TestWithParam<InstructionListTableStruct> {};

TEST_P(MLIR_InstructionListTableTest, instructionListTable) {
    const auto tables = GetParam();
    const auto range = tables.range;
    const auto shift = tables.shift;
    const auto bias = tables.bias;
    const auto expectedOutput = tables.expectedOutput;
    const auto outputBuffer = vpux::VPU::NCESparsity::getInstructionListTable(range, shift, bias);
    EXPECT_EQ(outputBuffer, expectedOutput);
}

std::vector<InstructionListTableStruct> instructionListsAndExpectedOutputs = {
        {/*range=*/{0, 255}, /*shift=*/{-5}, /*bias=*/{0},
         /*expectedOutput=*/{16386,     133710338, 133710850, 133711362, 133775362, 133775874, 133776386, 133776898,
                             133840898, -2473470,  -2472958,  -2472446,  -2408446,  -2407934,  -2407422,  6,
                             -2406910,  -2342910,  279042,    -2341886,  -2341374,  -2277374,  -2276862,  -2276350,
                             -2275838,  -2211838,  0,         6,         6,         6,         6,         6}},
        {/*range=*/{-5, -3, -1, 1, 254}, /*shift=*/{0, -1, 0, -1}, /*bias=*/{3, 4, 1, 0},
         /*expectedOutput=*/{-2605054,  -1555966, -506878, 542210,  133251074, 133251586, 133252098, 133252610,
                             133316610, 147970,   -375806, 148994,  -311294,   -310782,   -310270,   6,
                             -309758,   -245758,  1851906, 2376706, 804354,    344066,    -179710,   -179198,
                             -178686,   -114686,  0,       6,       6,         6,         6,         6}},
        {/*range=*/{-10, -8, -6, -4, 0, 253}, /*shift=*/{1, 0, 1, 1, 0}, /*bias=*/{3, 6, 2, 1, 0},
         /*expectedOutput=*/{-5226494,  -4177406, -3128318, -2079230, 81922,   132727298, 132727810, 132728322,
                             132792322, 672258,   148482,   673282,   737282,  213506,    214018,    6,
                             214530,    278530,   1851906,  3425282,  1328642, 868354,    344578,    345090,
                             345602,    409602,   0,        6,        6,       6,         6,         6}},
        {/*range=*/{-15, -13, -11, -9, -7, -5, -3, 0, 252}, /*shift=*/{2, 0, 2, 4, 2, 3, 1, 0},
         /*bias=*/{1, 10, 1, -1, 1, 0, 1, 0},
         /*expectedOutput=*/{-7847934,  -6798846, -5749758, -4700670, -3588094, -2539006, -1489918, 83458,
                             132268034, 1196546,  148482,   1197570,  2310146,  1262082,  1786882,  6,
                             738818,    278530,   803330,   5522434,  804354,   -180222,  868866,   345090,
                             869890,    409602,   0,        6,        6,        6,        6,        6}}};

INSTANTIATE_TEST_SUITE_P(Unit, MLIR_InstructionListTableTest, testing::ValuesIn(instructionListsAndExpectedOutputs));

}  // namespace
