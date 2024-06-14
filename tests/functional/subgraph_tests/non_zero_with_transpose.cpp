// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_ov2_layer_test.hpp"

#include <common/print_test_case_name.hpp>
#include <pretty_test_arguments.hpp>

#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/opsets/opset3.hpp>

using namespace ov::test;

namespace {

using ShapeAndOrder = std::pair<ov::test::InputShape, std::vector<int32_t>>;

using NonZeroWithTransposeTestParams = std::tuple<ShapeAndOrder, ov::element::Type>;

class NonZeroWithTransposeNPUTest :
        public testing::WithParamInterface<NonZeroWithTransposeTestParams>,
        public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::vector<int64_t>>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        const int32_t startFrom = 0;
        const int32_t range = 2;

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                        targetInputStaticShapes[i], range, startFrom);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    void SetUp() override {
        const auto& [shapeOrder, inputType] = this->GetParam();
        const auto& inputShape = shapeOrder.first;
        const auto& dimsOrder = shapeOrder.second;

        init_input_shapes({inputShape});
        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes) {
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inputType, shape));
        }

        auto nonZero = std::make_shared<ov::opset3::NonZero>(inputParams[0], ov::element::i64);
        inputParams[0]->set_friendly_name("input");

        auto convertI32 = std::make_shared<ov::op::v0::Convert>(nonZero, ov::element::i32);

        auto order = ov::op::v0::Constant::create(ov::element::i64, {dimsOrder.size()}, dimsOrder);
        auto transpose = std::make_shared<ov::op::v1::Transpose>(convertI32, order);

        auto results = ov::ResultVector();
        for (size_t i = 0; i < transpose->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::opset3::Result>(transpose->output(i)));
        }

        function = std::make_shared<ov::Model>(results, inputParams, "NonZero");
    }
};

// Tracking number: E#117210, E#119730
TEST_P(NonZeroWithTransposeNPUTest, DISABLED_NPU3720_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

const std::vector<ov::element::Type> inputPrecision = {ov::element::f32, ov::element::i32};
const std::vector<ShapeAndOrder> inShapes = {
        {staticShape(8, 32), {1, 0}}, {staticShape(8, 32), {0, 1}}, {staticShape(2, 88), {1, 0}}};

INSTANTIATE_TEST_SUITE_P(smoke_NonZeroWithTranspose, NonZeroWithTransposeNPUTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes), ::testing::ValuesIn(inputPrecision)),
                         PrintTestCaseName());
}  // namespace
