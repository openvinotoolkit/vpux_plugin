//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/cum_sum.hpp"
#include <vector>
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class CumSumLayerTestCommon : public CumSumLayerTest, virtual public VpuOv2LayerTest {};
class CumSumLayerTest_NPU3720 : public CumSumLayerTestCommon {};
class CumSumLayerTest_NPU4000 : public CumSumLayerTestCommon {};

TEST_P(CumSumLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(CumSumLayerTest_NPU4000, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {
const std::vector<std::vector<ov::Shape>> shapes = {{{5, 14, 5, 7}},
                                                    // Values from real neural networks
                                                    {{1, 1}},
                                                    {{1, 1024}},
                                                    {{1, 128}},
                                                    {{1, 25, 36}},
                                                    {{1, 384}},
                                                    {{1, 5}},
                                                    {{1, 9}},
                                                    {{8, 128}},
                                                    {{8, 384}}};

const std::vector<ov::element::Type> inputPrecision = {ov::element::f16, ov::element::f32};

const std::vector<int64_t> axes = {0, 1};
const std::vector<int64_t> negativeAxes = {-2, -1};

const std::vector<bool> exclusive = {true, false};
const std::vector<bool> reverse = {true, false};

const auto testCaseAxis_0 =
        testing::Combine(testing::ValuesIn({static_shapes_to_test_representation({shapes[0]})}),  // Input shapes
                         testing::Values(inputPrecision[0]),                                      // Model type
                         testing::Values(axes[0]),                                                // Axis
                         testing::Values(exclusive[0]),                                           // Exclusive
                         testing::Values(reverse[1]),                                             // Reverse
                         testing::Values(DEVICE_NPU));                                            // Device name

const auto testCasesNegativeAxis =
        testing::Combine(testing::ValuesIn({static_shapes_to_test_representation({shapes[0]})}),
                         testing::Values(inputPrecision[1]), testing::ValuesIn(negativeAxes),
                         testing::Values(exclusive[1]), testing::Values(reverse[0]), testing::Values(DEVICE_NPU));

std::vector<std::vector<ov::Shape>> iShape(shapes.begin() + 1, shapes.end());
const auto testCasesRealNet =
        testing::Combine(testing::ValuesIn(static_shapes_to_test_representation(iShape)),
                         testing::Values(inputPrecision[0]), testing::Values(axes[1]), testing::Values(exclusive[1]),
                         testing::Values(reverse[1]), testing::Values(DEVICE_NPU));

const auto testCasePrecommit =
        testing::Combine(testing::ValuesIn({static_shapes_to_test_representation({shapes[4]})}),
                         testing::Values(inputPrecision[0]), testing::Values(negativeAxes[0]),
                         testing::Values(exclusive[0]), testing::Values(reverse[0]), testing::Values(DEVICE_NPU));

// ------ NPU3720 ------
INSTANTIATE_TEST_SUITE_P(smoke_CumSum_axis_0, CumSumLayerTest_NPU3720, testCaseAxis_0,
                         CumSumLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CumSum_negative_axis, CumSumLayerTest_NPU3720, testCasesNegativeAxis,
                         CumSumLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CumSum_real_net, CumSumLayerTest_NPU3720, testCasesRealNet,
                         CumSumLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_CumSum, CumSumLayerTest_NPU3720, testCasePrecommit,
                         CumSumLayerTest_NPU3720::getTestCaseName);

// ------ NPU4000 ------
INSTANTIATE_TEST_SUITE_P(smoke_CumSum_axis_0, CumSumLayerTest_NPU4000, testCaseAxis_0,
                         CumSumLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CumSum_negative_axis, CumSumLayerTest_NPU4000, testCasesNegativeAxis,
                         CumSumLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CumSum_real_net, CumSumLayerTest_NPU4000, testCasesRealNet,
                         CumSumLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_CumSum, CumSumLayerTest_NPU4000, testCasePrecommit,
                         CumSumLayerTest_NPU4000::getTestCaseName);
}  // namespace
