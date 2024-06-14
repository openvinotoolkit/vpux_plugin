// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "common_test_utils/test_enums.hpp"

using namespace ov::test::utils;
namespace ov::test {

std::shared_ptr<ov::Node> build_activation_helper(const std::shared_ptr<ov::Node>& producer,
                                                  const ActivationTypes& act_type) {
    switch (act_type) {
    case ActivationTypes::None:
        return producer;
    case ActivationTypes::Relu:
        return std::make_shared<ov::op::v0::Relu>(producer->output(0));
    default:
        OPENVINO_THROW("build_activation_helper: unsupported activation type: ", act_type);
    }

    // execution must never reach this point
    return nullptr;
}

class Conv2dWithBiasTest_NPU3700 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<std::tuple<ov::Shape, ActivationTypes>> {
    void SetUp() override {
        const auto kenelShape = std::get<0>(GetParam());
        const size_t KERNEL_W = kenelShape.at(3);
        const size_t KERNEL_H = kenelShape.at(2);
        const size_t FILT_IN = kenelShape.at(1);
        const size_t FILT_OUT = kenelShape.at(0);
        const ov::Shape inputShape = {1, FILT_IN, KERNEL_H * 10, KERNEL_W * 10};
        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        std::vector<float> weights(FILT_IN * FILT_OUT * KERNEL_W * KERNEL_H);
        for (std::size_t i = 0; i < weights.size(); i++) {
            weights.at(i) = std::cos(i * 3.14 / 6);
        }
        auto constLayer_node = std::make_shared<ov::op::v0::Constant>(
                ov::element::Type_t::f32, ov::Shape{FILT_OUT, FILT_IN, KERNEL_H, KERNEL_W}, weights.data());

        auto conv2d_node = std::make_shared<ov::op::v1::Convolution>(
                params.at(0), constLayer_node->output(0), ov::Strides(std::vector<size_t>{1, 1}),
                ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                ov::Strides(std::vector<size_t>{1, 1}));

        std::vector<float> biases(FILT_OUT, 1.0);
        for (std::size_t i = 0; i < biases.size(); i++) {
            biases.at(i) = i * 0.25f;
        }
        auto bias_weights_node = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32,
                                                                        ov::Shape{1, FILT_OUT, 1, 1}, biases.data());

        auto bias_node = std::make_shared<ov::op::v1::Add>(conv2d_node->output(0), bias_weights_node->output(0));
        auto act_type = std::get<1>(GetParam());
        auto act_node = build_activation_helper(bias_node, act_type);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(act_node)};

        function = std::make_shared<ov::Model>(results, params, "Conv2dWithBiasTest");
        rel_threshold = 0.5f;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<ov::Shape, ActivationTypes>>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_P(Conv2dWithBiasTest_NPU3700, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3700);
}

TEST_P(Conv2dWithBiasTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

const std::vector<ov::Shape> kernelShapes = {
        {11, 3, 2, 2},
        {16, 3, 2, 2},
        {11, 16, 2, 2},
        {16, 16, 2, 2},
};

const std::vector<ActivationTypes> activations = {
        ActivationTypes::None,
        ActivationTypes::Relu,
};

INSTANTIATE_TEST_SUITE_P(conv2d_with_act, Conv2dWithBiasTest_NPU3700,
                         ::testing::Combine(::testing::ValuesIn(kernelShapes), ::testing::ValuesIn(activations)),
                         Conv2dWithBiasTest_NPU3700::getTestCaseName);

}  // namespace ov::test
