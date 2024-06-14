// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov::test {

enum class PostOp { SIGMOID, TANH, PRELU };

class ConvPwlSubGraphTest_NPU3700 : public VpuOv2LayerTest, public testing::WithParamInterface<PostOp> {
    void SetUp() override {
        const ov::Shape inputShape{1, 3, 32, 32};
        const ov::Shape weightsShape{16, 3, 1, 1};

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(inputShape))};

        const auto weightsU8 =
                ov::test::utils::deprecated::make_constant<uint8_t>(ov::element::u8, weightsShape, {}, true, 255, 0);
        const auto weightsFP32 = std::make_shared<ov::op::v0::Convert>(weightsU8, ov::element::f32);

        const ov::Strides strides = {1, 1};
        const ov::CoordinateDiff pads_begin = {0, 0};
        const ov::CoordinateDiff pads_end = {0, 0};
        const ov::Strides dilations = {1, 1};
        const auto conv = std::make_shared<ov::op::v1::Convolution>(params[0], weightsFP32, strides, pads_begin,
                                                                    pads_end, dilations);

        std::shared_ptr<ov::Node> postOp;
        auto postOpType = GetParam();
        if (postOpType == PostOp::SIGMOID) {
            postOp = std::make_shared<ov::op::v0::Sigmoid>(conv);
        } else if (postOpType == PostOp::TANH) {
            postOp = std::make_shared<ov::op::v0::Tanh>(conv);
        } else if (postOpType == PostOp::PRELU) {
            const auto negativeSlope =
                    ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, std::vector<float>{0.1f});
            postOp = std::make_shared<ov::op::v0::PRelu>(conv, negativeSlope);
        }

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(postOp)};
        function = std::make_shared<ov::Model>(results, params, "ConvPwl");
        rel_threshold = 0.1f;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<PostOp>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

class ConvPwlQuantizedSubGraphTest_NPU3700 : public VpuOv2LayerTest, public testing::WithParamInterface<PostOp> {
    void SetUp() override {
        const ov::Shape inputShape{1, 3, 32, 32};
        const ov::Shape weightsShape{16, 3, 1, 1};

        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        const size_t dataLevels = 256;
        const auto dataFq = ov::test::utils::make_fake_quantize(params[0], ov::element::f32, dataLevels, {}, {-3.0},
                                                                {3.0}, {-3.0}, {3.0});

        const auto weightsU8 =
                ov::test::utils::deprecated::make_constant<uint8_t>(ov::element::u8, weightsShape, {}, true, 255, 0);
        const auto weightsFP32 = std::make_shared<ov::op::v0::Convert>(weightsU8, ov::element::f32);

        const auto weightsInLow =
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, std::vector<float>{0.0f});
        const auto weightsInHigh =
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, std::vector<float>{255.0f});
        std::vector<float> perChannelLow(weightsShape[0], 0.0f);
        std::vector<float> perChannelHigh(weightsShape[0], 1.0f);
        const auto weightsOutLow =
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{weightsShape[0], 1, 1, 1}, perChannelLow);
        const auto weightsOutHigh =
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{weightsShape[0], 1, 1, 1}, perChannelHigh);

        const size_t weightsLevels = 256;
        const auto weightsFq = std::make_shared<ov::op::v0::FakeQuantize>(weightsFP32, weightsInLow, weightsInHigh,
                                                                          weightsOutLow, weightsOutHigh, weightsLevels);

        const ov::Strides strides = {1, 1};
        const ov::CoordinateDiff pads_begin = {0, 0};
        const ov::CoordinateDiff pads_end = {0, 0};
        const ov::Strides dilations = {1, 1};
        const auto conv =
                std::make_shared<ov::op::v1::Convolution>(dataFq, weightsFq, strides, pads_begin, pads_end, dilations);

        std::shared_ptr<ov::Node> outputFq;
        const size_t outLevels = 256;
        auto postOpType = GetParam();
        if (postOpType == PostOp::SIGMOID) {
            const auto postOp = std::make_shared<ov::op::v0::Sigmoid>(conv);
            outputFq = ov::test::utils::make_fake_quantize(postOp, ov::element::f32, outLevels, {}, {0.0}, {1.0}, {0.0},
                                                           {1.0});
        } else if (postOpType == PostOp::TANH) {
            const auto postOp = std::make_shared<ov::op::v0::Tanh>(conv);
            outputFq = ov::test::utils::make_fake_quantize(postOp, ov::element::f32, outLevels, {}, {-1.0}, {1.0},
                                                           {-1.0}, {1.0});
        } else if (postOpType == PostOp::PRELU) {
            const auto negativeSlope =
                    ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, std::vector<float>{0.1f});
            const auto postOp = std::make_shared<ov::op::v0::PRelu>(conv, negativeSlope);
            outputFq = ov::test::utils::make_fake_quantize(postOp, ov::element::f32, outLevels, {}, {-128.0}, {127.0},
                                                           {-128.0}, {127.0});
        }

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(outputFq)};
        function = std::make_shared<ov::Model>(results, params, "ConvPwlQuantized");
        rel_threshold = 0.1f;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<PostOp>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_P(ConvPwlSubGraphTest_NPU3700, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3700);
}

TEST_P(ConvPwlSubGraphTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

TEST_P(ConvPwlQuantizedSubGraphTest_NPU3700, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3700);
}

TEST_P(ConvPwlQuantizedSubGraphTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

std::vector<PostOp> postOps = {PostOp::SIGMOID, PostOp::TANH, PostOp::PRELU};

// TODO: investigate bad accuracy for both SW and HW
// prelu quantized test
std::vector<PostOp> quantPostOps = {
        PostOp::SIGMOID, PostOp::TANH
        //, PostOp::PRELU
};

INSTANTIATE_TEST_CASE_P(smoke_ConvPwl, ConvPwlSubGraphTest_NPU3700, ::testing::ValuesIn(postOps),
                        ConvPwlSubGraphTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ConvPwlQuantized, ConvPwlQuantizedSubGraphTest_NPU3700, ::testing::ValuesIn(quantPostOps),
                        ConvPwlQuantizedSubGraphTest_NPU3700::getTestCaseName);

}  // namespace ov::test
