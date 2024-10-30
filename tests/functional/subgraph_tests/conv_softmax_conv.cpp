// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

namespace ov::test {

struct ConvSoftmaxConvTestParams {
    ov::Shape inputShape;
    int64_t axis;
    size_t softmaxChannelSize;
};

class ConvSoftmaxConvTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<ConvSoftmaxConvTestParams> {
    void SetUp() override {
        inType = ov::element::f16;
        outType = ov::element::f16;
        const auto testParams = GetParam();
        int64_t axis = testParams.axis;
        const auto inputShape = testParams.inputShape;
        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape}));
        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.front())};
        const auto conv1 = buildConv(params.at(0), testParams.softmaxChannelSize);
        const auto softMax = std::make_shared<ov::op::v1::Softmax>(conv1->output(0), axis);
        const auto conv2 = buildConv(softMax->output(0), inputShape.at(1));
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(conv2)};
        function = std::make_shared<ov::Model>(results, params, "ConvSoftmaxConvTest");
        abs_threshold = 0.008;
    }

    std::shared_ptr<ov::Node> buildConv(const ov::Output<ov::Node>& param, size_t softmaxChannelSize) {
        const ov::Shape inputShape = param.get_shape();
        const auto weightsSize = inputShape.at(1) * softmaxChannelSize * 1 * 1;
        std::vector<float> values(weightsSize, 0.0f);
        for (std::size_t i = 0; i < softmaxChannelSize; i++) {
            values.at(i * inputShape.at(1) + i % inputShape.at(1)) = 1.0f;
        }
        const auto weightsShape = ov::Shape{softmaxChannelSize, inputShape.at(1), 1, 1};
        const auto weights = ov::op::v0::Constant::create(ov::element::f16, weightsShape, values);
        auto conv2d_node = std::make_shared<ov::op::v1::Convolution>(
                param, weights->output(0), ov::Strides(std::vector<size_t>{1, 1}),
                ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                ov::Strides(std::vector<size_t>{1, 1}));

        return conv2d_node;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvSoftmaxConvTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_P(ConvSoftmaxConvTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ConvSoftmaxConvTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

INSTANTIATE_TEST_SUITE_P(smoke_ConvSoftmaxConv, ConvSoftmaxConvTestCommon,
                         ::testing::Values(ConvSoftmaxConvTestParams{{1, 48, 8, 16}, 1, 77},
                                           ConvSoftmaxConvTestParams{{1, 48, 8, 16}, 1, 21}),
                         ConvSoftmaxConvTestCommon::getTestCaseName);

}  // namespace ov::test
