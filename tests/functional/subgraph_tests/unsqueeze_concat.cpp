//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

class UnsqueezeConcat : public testing::WithParamInterface<std::string>, public VpuOv2LayerTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        const auto& funcInputs = function->inputs();
        for (size_t inIdx = 0; inIdx < funcInputs.size(); inIdx++) {
            const auto& inputStaticShape = inputShapes[inIdx];
            const auto totalSize = ov::shape_size(inputStaticShape);
            auto inputTensor = ov::Tensor{ov::element::f16, inputStaticShape};
            auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f16>::value_type>();
            for (size_t i = 0; i < totalSize; i++) {
                inputData[i] = 1 + inIdx + i % 3;
            }
            inputs[funcInputs[inIdx].get_node_shared_ptr()] = inputTensor;
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
        const ov::PartialShape dynamicShape{1, ov::Dimension{1, 64}, 128};
        const ov::Shape staticShape{1, 32, 128};
        const std::vector<ov::Shape> inferenceShapes = {staticShape};
        const ov::test::InputShape dataShape = {dynamicShape, inferenceShapes};
        init_input_shapes({dataShape, dataShape});
        const auto lhsParam = std::make_shared<ov::opset1::Parameter>(ov::element::f16, inputDynamicShapes.at(0));
        const auto rhsParam = std::make_shared<ov::opset1::Parameter>(ov::element::f16, inputDynamicShapes.at(1));

        const auto dims = ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
        const auto lhsUnsqueeze = std::make_shared<ov::opset1::Unsqueeze>(lhsParam->output(0), dims->output(0));
        const auto rhsUnsqueeze = std::make_shared<ov::opset1::Unsqueeze>(rhsParam->output(0), dims->output(0));
        const auto concat = std::make_shared<ov::opset1::Concat>(
                ov::OutputVector{
                        lhsUnsqueeze->output(0),
                        rhsUnsqueeze->output(0),
                },
                1);

        const auto order =
                ov::opset1::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{2, 0, 1, 3});
        const auto transpose = std::make_shared<ov::opset1::Transpose>(concat->output(0), order->output(0));

        const auto outputShape =
                ov::opset1::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{-1, 1, 256});
        const auto reshape = std::make_shared<ov::opset1::Reshape>(transpose->output(0), outputShape->output(0), true);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(reshape->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{lhsParam, rhsParam}, "MatMulWithFQ");
    }
};

//
// Platform test definition
//

TEST_P(UnsqueezeConcat, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    configuration["NPU_COMPILATION_MODE_PARAMS"] = GetParam();
    run(Platform::NPU3720);
}

TEST_P(UnsqueezeConcat, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    configuration["NPU_COMPILATION_MODE_PARAMS"] = GetParam();
    run(Platform::NPU4000);
}

const std::vector<std::string> compilationModeParams = {
        "enable-extra-shape-bound-ops=false",
        "enable-extra-shape-bound-ops=true",
};

INSTANTIATE_TEST_SUITE_P(DynamicUnsqueezeConcat, UnsqueezeConcat, ::testing::ValuesIn(compilationModeParams));

}  // namespace ov::test::subgraph
