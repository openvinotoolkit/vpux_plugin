//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>
#include "shared_test_classes/single_op/concat.hpp"

using namespace ov::test::utils;
using namespace ov::test;

namespace ConcatSoftmaxSubGraphTestsDefinitions {
class ConcatSoftmaxSubGraphTest_NPU3700 : public VpuOv2LayerTest, public ConcatLayerTest {
    void SetUp() override {
        int axis;
        std::vector<InputShape> inputShape;
        std::tie(axis, inputShape, inType, targetDevice) = this->GetParam();

        init_input_shapes(inputShape);

        ov::ParameterVector params;
        ov::NodeVector paramsNodes;
        for (const auto& shape : inputDynamicShapes) {
            auto param = std::make_shared<ov::op::v0::Parameter>(inType, shape);
            params.push_back(param);
            paramsNodes.push_back(param);
        }
        auto concat = std::make_shared<ov::op::v0::Concat>(paramsNodes, axis);

        auto softMax = std::make_shared<ov::op::v1::Softmax>(concat, axis);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(softMax)};
        function = std::make_shared<ov::Model>(results, params, "concat_softmax");

        rel_threshold = 0.1f;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<concatParamsTuple>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << ConcatLayerTest::getTestCaseName(obj) << sep;

        return result.str();
    }
};

TEST_P(ConcatSoftmaxSubGraphTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16,
};

// Note: npu-plugin does not support batch-size > 1.

// 4d cases
std::vector<int> axes4d = {1, 2, 3};
std::vector<std::vector<ov::Shape>> inShapes4d = {
        {{1, 2, 3, 4}, {1, 2, 3, 4}, {1, 2, 3, 4}},
        {{1, 10, 10, 10}, {1, 10, 10, 10}, {1, 10, 10, 10}},
        {{1, 10, 33, 80}, {1, 10, 33, 80}, {1, 10, 33, 80}, {1, 10, 33, 80}},
};

INSTANTIATE_TEST_SUITE_P(smoke4d_tensors, ConcatSoftmaxSubGraphTest_NPU3700,
                         ::testing::Combine(::testing::ValuesIn(axes4d),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inShapes4d)),
                                            ::testing::ValuesIn(netPrecisions), ::testing::Values(DEVICE_NPU)),
                         ConcatSoftmaxSubGraphTest_NPU3700::getTestCaseName);

// 3d cases
std::vector<int> axes3d = {1, 2};
std::vector<std::vector<ov::Shape>> inShapes3d = {
        {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}},
        {{1, 10, 33}, {1, 10, 33}, {1, 10, 33}, {1, 10, 33}},
};

INSTANTIATE_TEST_SUITE_P(smoke3d_tensors, ConcatSoftmaxSubGraphTest_NPU3700,
                         ::testing::Combine(::testing::ValuesIn(axes3d),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inShapes4d)),
                                            ::testing::ValuesIn(netPrecisions), ::testing::Values(DEVICE_NPU)),
                         ConcatSoftmaxSubGraphTest_NPU3700::getTestCaseName);

// Check parameters from squeezenet1_1
std::vector<int> axes_squeeznet1_1 = {1};

std::vector<std::vector<ov::Shape>> inShapes_squeeznet1_1 = {{{1, 64, 56, 56}, {1, 64, 56, 56}},
                                                             {{1, 192, 14, 14}, {1, 192, 14, 14}},
                                                             {{1, 128, 28, 28}, {1, 128, 28, 28}},
                                                             {{1, 256, 14, 14}, {1, 256, 14, 14}}};

INSTANTIATE_TEST_SUITE_P(
        smoke_squeeznet1_1_tensors, ConcatSoftmaxSubGraphTest_NPU3700,
        ::testing::Combine(::testing::ValuesIn(axes_squeeznet1_1),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_squeeznet1_1)),
                           ::testing::ValuesIn(netPrecisions), ::testing::Values(DEVICE_NPU)),
        ConcatSoftmaxSubGraphTest_NPU3700::getTestCaseName);

}  // namespace ConcatSoftmaxSubGraphTestsDefinitions
