//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/shape.hpp>
#include "common/npu_test_env_cfg.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"
#include "vpux/utils/core/error.hpp"

namespace ov::test::subgraph {

using DynamicSigmoidWithShapeOfConfig = std::tuple<std::vector<int64_t>>;

class NPUDynamicSigmoidWithShapeOfTest :
        public testing::WithParamInterface<DynamicSigmoidWithShapeOfConfig>,
        public VpuOv2LayerTest {
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

        const auto& expected = expectedTensors[0];
        const auto& actual = actualTensors[0];
        ASSERT_EQ(expected.get_size(), actual.get_size());

        const float absThreshold = std::numeric_limits<float>::epsilon();
        const float relThreshold = std::numeric_limits<float>::epsilon();
        ov::test::utils::compare(expected, actual, absThreshold, relThreshold);
    }

    void SetUp() override {
        const auto& inferenceShapeValues = std::get<0>(GetParam());
        ov::Shape inferenceShape(inferenceShapeValues.begin(), inferenceShapeValues.end());

        const auto dataShape = std::make_pair(ov::PartialShape{1, 3, ov::Dimension(1, 10), ov::Dimension(1, 16)},
                                              std::vector<ov::Shape>{inferenceShape});

        init_input_shapes({dataShape});

        const auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, inputDynamicShapes.at(0));
        const auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(param->output(0));

        const auto results = std::make_shared<ov::opset1::ShapeOf>(sigmoid->output(0));
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "DynamicSigmoidWithShapeOf");
    }
};

//
// Platform test definition
//

TEST_P(NPUDynamicSigmoidWithShapeOfTest, NPU3720_HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    setMLIRCompilerType();
    run(Platform::NPU3720);
}

const std::vector<std::vector<int64_t>> inferenceShapes = {{1, 3, 5, 8}, {1, 3, 8, 15}};

INSTANTIATE_TEST_SUITE_P(DynamicSigmoidWithShapeOf, NPUDynamicSigmoidWithShapeOfTest,
                         ::testing::Combine(::testing::ValuesIn(inferenceShapes)));

}  // namespace ov::test::subgraph
