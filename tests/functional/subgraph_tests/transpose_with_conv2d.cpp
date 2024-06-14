// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

namespace ov::test {

class TransposeWithConv2dTest_NPU3700 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<std::vector<int64_t>> {
    void SetUp() override {
        const ov::Shape inputShape = {1, 16, 32, 64};

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        const std::vector<int64_t> transposeArg0Weights = GetParam();
        auto transposeArg0Const = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{transposeArg0Weights.size()}, transposeArg0Weights.data());

        auto transposeArg0Node = std::make_shared<ov::op::v1::Transpose>(params.at(0), transposeArg0Const->output(0));

        const size_t planesOut = 16;
        const size_t planesIn = transposeArg0Node->get_shape().at(1);
        const size_t kernelY = 3;
        const size_t kernelX = 3;

        std::vector<float> weights(planesIn * planesOut * kernelY * kernelX);
        for (std::size_t i = 0; i < weights.size(); i++) {
            weights.at(i) = std::cos(i * 3.14 / 6.f);
        }
        auto constLayerNode = std::make_shared<ov::op::v0::Constant>(
                ov::element::f32, ov::Shape{planesOut, planesIn, kernelY, kernelX}, weights.data());

        auto conv2dNode = std::make_shared<ov::op::v1::Convolution>(
                transposeArg0Node->output(0), constLayerNode->output(0), ov::Strides(std::vector<size_t>{1, 1}),
                ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                ov::Strides(std::vector<size_t>{1, 1}));

        auto reluNode = std::make_shared<ov::op::v0::Relu>(conv2dNode->output(0));
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(reluNode)};

        function = std::make_shared<ov::Model>(results, params, "TransposeWithConv2dTest");
        rel_threshold = 0.5f;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::vector<int64_t>>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_P(TransposeWithConv2dTest_NPU3700, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3700);
}

TEST_P(TransposeWithConv2dTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

const std::vector<std::vector<int64_t>> transposes = {
        {0, 3, 1, 2}, {0, 3, 2, 1}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 1, 3, 2},
};

INSTANTIATE_TEST_SUITE_P(transpose_conv2d, TransposeWithConv2dTest_NPU3700, ::testing::ValuesIn(transposes),
                         TransposeWithConv2dTest_NPU3700::getTestCaseName);

}  // namespace ov::test
