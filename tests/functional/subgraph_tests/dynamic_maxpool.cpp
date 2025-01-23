//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

class DynamicMaxPoolTest : public VpuOv2LayerTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        OPENVINO_ASSERT(inputShapes.size() == 1, "Only 1 input shape is supported");
        const auto& funcInputs = function->inputs();
        OPENVINO_ASSERT(funcInputs.size() == 1, "Only 1 input is supported");
        const auto& inputStaticShape = inputShapes[0];
        const auto totalSize = ov::shape_size(inputStaticShape);
        auto inputTensor = ov::Tensor{ov::element::f16, inputStaticShape};
        auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f16>::value_type>();
        for (size_t i = 0; i < totalSize; i++) {
            inputData[i] = i % 17;
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

        const float absThreshold = 0.125f;
        ov::test::utils::compare(actual, expected, absThreshold);
    }

    void SetUp() override {
        const ov::PartialShape inDynShape = {1, 3, 16, ov::Dimension(1, 32)};
        const std::vector<ov::Shape> inferenceShapes = {ov::Shape{1, 3, 16, 21}};
        const ov::test::InputShape dataShape = {inDynShape, inferenceShapes};
        init_input_shapes({dataShape});
        const auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f16, inputDynamicShapes.at(0));

        const ov::Strides strides = {2, 2};
        const ov::Shape padsBegin = {0, 0};
        const ov::Shape padsEnd = {0, 0};
        const ov::Shape kernel = {2, 2};
        const auto maxpool = std::make_shared<ov::op::v1::MaxPool>(param, strides, padsBegin, padsEnd, kernel);
        const auto relu = std::make_shared<ov::opset1::Relu>(maxpool->output(0));

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(relu->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "DynamicMaxPool");
    }
};

//
// Platform test definition
//

TEST_F(DynamicMaxPoolTest, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(DynamicMaxPoolTest, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
}  // namespace ov::test::subgraph
