//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/concat.hpp"
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov2_layer_test.hpp"

#include <vector>

using namespace ov::test::utils;

namespace ov {
namespace test {

class ConcatLayerTestCommon : public ConcatLayerTest, virtual public VpuOv2LayerTest {};

class ConcatLayerTest_NPU3720 : public ConcatLayerTestCommon {};
class ConcatLayerTest_NPU4000 : public ConcatLayerTestCommon {};

TEST_P(ConcatLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ConcatLayerTest_NPU4000, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

std::vector<int> axes = {0, 1, 2, 3};

// ------ NPU3720/4000 ------

const auto concatParams = ::testing::Combine(
        ::testing::ValuesIn(axes),
        ::testing::Values(ov::test::static_shapes_to_test_representation({{1, 16, 10, 10}, {1, 16, 10, 10}})),
        ::testing::Values(ov::element::u8), ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Concat, ConcatLayerTest_NPU3720, concatParams,
                         ConcatLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Concat, ConcatLayerTest_NPU4000, concatParams,
                         ConcatLayerTest_NPU4000::getTestCaseName);

}  // namespace
