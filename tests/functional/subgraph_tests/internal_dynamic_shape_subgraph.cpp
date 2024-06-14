// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_ov2_layer_test.hpp"

#include <shared_test_classes/base/ov_subgraph.hpp>

#include <common/functions.h>
#include <common/print_test_case_name.hpp>
#include <pretty_test_arguments.hpp>

#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset4.hpp>

using namespace ov::test;
using namespace ov::test::utils;

namespace {

class InternalDynamicShapesNPUTest : public testing::WithParamInterface<ov::test::InputShape>, public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ov::test::InputShape>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;

        result << "IS=" << vec2str(obj.param.second) << sep;

        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        const int32_t startFrom = 0;
        const int32_t range = 2;

        for (const auto& funcInput : funcInputs) {
            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                        targetInputStaticShapes[0], range, startFrom);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

protected:
    void SetUp() override {
        const auto& inputShape = this->GetParam();
        const std::vector<int64_t> dimsOrder = {1, 0};

        init_input_shapes({inputShape});
        ov::ParameterVector inputParams;

        for (auto&& shape : inputDynamicShapes) {
            // nonZero input
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::i32, shape));
            // scatterNDUpdate input
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape));
        }

        auto nonZero = std::make_shared<ov::opset3::NonZero>(inputParams[0], ov::element::i64);
        inputParams[0]->set_friendly_name("input_0");

        auto convertI32 = std::make_shared<ov::op::v0::Convert>(nonZero, ov::element::i32);

        auto order = ov::op::v0::Constant::create(ov::element::i64, {dimsOrder.size()}, dimsOrder);
        auto transpose = std::make_shared<ov::op::v1::Transpose>(convertI32, order);

        auto shapeOf = std::make_shared<ov::opset3::ShapeOf>(transpose);

        auto gather =
                std::make_shared<ov::opset3::Gather>(shapeOf, ov::opset3::Constant::create(ov::element::i64, {1}, {0}),
                                                     ov::opset3::Constant::create(ov::element::i64, {1}, {0}));

        // Values have been taken from a real model
        auto data = std::make_shared<ov::opset3::Constant>(ov::element::f32, ov::Shape{9},
                                                           std::vector<float>{1, 0, 0, 0, -1, 0, 0, 0, -1});
        auto stridedSlice = std::make_shared<ov::opset3::StridedSlice>(
                data, ov::opset3::Constant::create(ov::element::i64, {1}, {0}), gather,
                ov::opset3::Constant::create(ov::element::i64, {1}, {1}), std::vector<int64_t>{0},
                std::vector<int64_t>{0});

        inputParams[1]->set_friendly_name("input_1");
        auto scatterNDUpdate = std::make_shared<ov::opset4::ScatterNDUpdate>(inputParams[1], transpose, stridedSlice);

        auto results = ov::ResultVector();
        for (size_t i = 0; i < transpose->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::opset3::Result>(scatterNDUpdate->output(i)));
        }

        function = std::make_shared<ov::Model>(results, inputParams, "DynamicShapeSubgraph");
    }
};

TEST_P(InternalDynamicShapesNPUTest, NPU3720_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

const std::vector<ov::test::InputShape> inShapes = {staticShape(1, 3, 3)};

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_InternalDynamicShapes, InternalDynamicShapesNPUTest,
                         ::testing::ValuesIn(inShapes), InternalDynamicShapesNPUTest::getTestCaseName);
}  // namespace
