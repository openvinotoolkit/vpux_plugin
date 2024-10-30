//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"

#include "vpux/utils/core/error.hpp"
namespace ov::test::subgraph {

class Conv2dInMixedMode : public VpuOv2LayerTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        VPUX_THROW_UNLESS(inputShapes.size() == 1, "Only 1 input shape is supported");
        const auto& funcInputs = function->inputs();
        VPUX_THROW_UNLESS(funcInputs.size() == 1, "Only 1 input is supported");
        const auto& inputStaticShape = inputShapes[0];
        const auto totalSize = ov::shape_size(inputStaticShape);
        auto inputTensor = ov::Tensor{ov::element::f32, inputStaticShape};
        auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f32>::value_type>();
        for (size_t i = 0; i < totalSize; i++) {
            inputData[i] = 1 + i % 7;
        }
        inputs = {
                {funcInputs[0].get_node_shared_ptr(), inputTensor},
        };
    }

    void compare(const std::vector<ov::Tensor>& expectedTensors,
                 const std::vector<ov::Tensor>& actualTensors) override {
        ASSERT_EQ(actualTensors.size(), 1);
        ASSERT_EQ(expectedTensors.size(), 1);

        const auto expected = expectedTensors[0];
        const auto actual = actualTensors[0];
        ASSERT_EQ(expected.get_size(), actual.get_size());

        const float absThreshold = 0.5f;
        ov::test::utils::compare(expected, actual, absThreshold);
    }

    void SetUp() override {
        const size_t FILT_IN = 16;
        const ov::Shape inputShape = {1, FILT_IN, 1, 1};
        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        const size_t FILT_OUT = 16;
        const size_t KERNEL_H = 1;
        const size_t KERNEL_W = 1;
        const ov::Shape weightShape = ov::Shape{FILT_OUT, FILT_IN, KERNEL_H, KERNEL_W};
        std::vector<uint8_t> weightsData(FILT_IN * FILT_OUT * KERNEL_W * KERNEL_H);
        for (size_t i = 0; i < weightsData.size(); i++) {
            weightsData.at(i) = 129 + i % 3;
        }
        weightsData.at(0) = 0;
        weightsData.at(1) = 255;
        const auto constLayer_node = ov::opset1::Constant::create(ov::element::u8, weightShape, weightsData);
        const float zp = 128.f;

        auto convert_node = std::make_shared<ov::opset1::Convert>(constLayer_node->output(0), ov::element::f32);

        const auto zero_points = ov::opset1::Constant::create(ov::element::f32, {1}, std::vector<float>{zp});
        const auto shift_node = std::make_shared<ov::opset1::Subtract>(convert_node->output(0), zero_points->output(0));

        const auto scales = ov::opset1::Constant::create(ov::element::f32, {1}, std::vector<float>{2.f});
        const auto scale_node = std::make_shared<ov::opset1::Multiply>(shift_node->output(0), scales->output(0));

        auto conv2d_node = std::make_shared<ov::op::v1::Convolution>(
                params.at(0), scale_node->output(0), ov::Strides(std::vector<size_t>{1, 1}),
                ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                ov::Strides(std::vector<size_t>{1, 1}));

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(conv2d_node)};

        function = std::make_shared<ov::Model>(results, params, "Conv2dInMixedMode");
        rel_threshold = 0.5f;
    }
};

//
// Platform test definition
//

TEST_F(Conv2dInMixedMode, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(Conv2dInMixedMode, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
}  // namespace ov::test::subgraph
