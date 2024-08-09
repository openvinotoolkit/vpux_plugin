//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

class TwoMatMulsWithAlignment : public VpuOv2LayerTest {
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
            inputData[i] = std::sin(i);
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
        ov::test::utils::compare(actual, expected, absThreshold);
    }

    std::shared_ptr<ov::Node> buildMatMul(const ov::Output<ov::Node>& value, const ov::Shape& weightShape,
                                          const bool transposeA) {
        const auto weightTotalSize1 = ov::shape_size(weightShape);
        std::vector<ov::float16> weightsData(weightTotalSize1, 0);
        for (size_t i = 0; i < weightsData.size(); i++) {
            weightsData.at(i) = std::floor(30.f * std::cos(i));
        }
        const auto weights = ov::opset1::Constant::create(ov::element::f16, weightShape, weightsData);
        const auto convert = std::make_shared<ov::opset1::Convert>(weights->output(0), ov::element::f32);
        return std::make_shared<ov::opset1::MatMul>(value, convert->output(0), transposeA, true);
    }

    void SetUp() override {
        const ov::Shape lhsShape = {1, 2, 128, 512};
        const std::vector<ov::Shape> inferenceShapes = {lhsShape};
        const ov::test::InputShape dataShape = {lhsShape, inferenceShapes};
        init_input_shapes({dataShape});
        const auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, inputDynamicShapes.at(0));
        const ov::Shape weightShape1 = {1, 1, 9, 128};
        const auto matmul1 = buildMatMul(param->output(0), weightShape1, true);
        const ov::Shape weightShape2 = {1, 1, 128, 9};
        const auto matmul2 = buildMatMul(matmul1->output(0), weightShape2, false);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(matmul2->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "MatMulWithAlign");

        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_element_type(ov::element::f16);
        preProc.output().tensor().set_element_type(ov::element::f16);
        function = preProc.build();
    }
};

TEST_F(TwoMatMulsWithAlignment, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_F(TwoMatMulsWithAlignment, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace ov::test::subgraph
