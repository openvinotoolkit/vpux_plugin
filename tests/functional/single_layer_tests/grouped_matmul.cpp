//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

class GroupedMatMulTest : public VpuOv2LayerTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        OPENVINO_ASSERT(inputShapes.size() == 1, "Only 1 input shape is supported");
        const auto& funcInputs = function->inputs();
        OPENVINO_ASSERT(funcInputs.size() == 1, "Only 1 input is supported");
        inputs.clear();
        const int32_t startFrom = 0;
        const int32_t range = 10;
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), inputShapes[i],
                                                                        range, startFrom);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
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

    void SetUp() override {
        const ov::Shape lhsShape = {1, 25, 8, 64};
        const std::vector<ov::Shape> inferenceShapes = {lhsShape};
        const ov::test::InputShape dataShape = {lhsShape, inferenceShapes};
        init_input_shapes({dataShape});
        const auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f16, inputDynamicShapes.at(0));
        const ov::Shape weightShape = {1, 25, 16, 64};
        const auto weightTotalSize = ov::shape_size(weightShape);
        std::vector<ov::float16> weightsData(weightTotalSize, 0);
        for (size_t i = 0; i < weightsData.size(); i++) {
            weightsData.at(i) = i % 3;
        }
        const auto weights = ov::opset1::Constant::create(ov::element::f16, weightShape, weightsData);

        const auto matmul = std::make_shared<ov::opset1::MatMul>(param->output(0), weights->output(0), false, true);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(matmul->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "GroupedMatMul");
    }
};

//
// Platform test definition
//

TEST_F(GroupedMatMulTest, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    configuration["NPU_COMPILATION_MODE_PARAMS"] =
            "enable-grouped-matmul=true enable-weights-swizzling=true enable-activation-swizzling=true";
    run(Platform::NPU4000);
}

}  // namespace ov::test::subgraph
