//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/const/utils/sub_byte.hpp"
#include <gtest/gtest.h>

using namespace vpux;

struct SubByteUnpackingParams {
    mlir::SmallVector<char> inputVector;
    size_t bitWidth;
    mlir::SmallVector<char> expectedOutputVector;
};

class SubByteUnpackingTests : public testing::TestWithParam<SubByteUnpackingParams> {};

TEST_P(SubByteUnpackingTests, getConstBuffer) {
    const auto params = GetParam();
    const auto numElems = params.bitWidth < CHAR_BIT ? params.inputVector.size() * CHAR_BIT / params.bitWidth
                                                     : params.inputVector.size();
    const auto actualData = vpux::Const::getConstBuffer(params.inputVector.data(), params.bitWidth, numElems);
    EXPECT_EQ(actualData, params.expectedOutputVector);
}

// clang-format off
std::vector<SubByteUnpackingParams> getConstBufferParams = {
        {/*inputVector*/ {0x01, 0x23}, /*bitWidth*/ 16, /*outputVector*/ {0x01, 0x23}},
        {/*inputVector*/ {0x01, 0x23}, /*bitWidth*/ 8, /*outputVector*/ {0x01, 0x23}},
        {/*inputVector*/ {0x10, 0x32}, /*bitWidth*/ 4, /*outputVector*/ {0x0, 0x1, 0x2, 0x3}},
        {/*inputVector*/ {-0x1C, 0x1B}, /*bitWidth*/ 2, /*outputVector*/ {0x0, 0x1, 0x2, 0x3, 0x3, 0x2, 0x1, 0x0}},
};
// clang-format on

INSTANTIATE_TEST_CASE_P(Unit, SubByteUnpackingTests, testing::ValuesIn(getConstBufferParams));
