//
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
#include <openvino/op/broadcast.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset4.hpp>
#include <openvino/opsets/opset5.hpp>
#include <openvino/pass/manager.hpp>

#include <transformations/op_conversions/bidirectional_sequences_decomposition.hpp>
#include <transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp>

using namespace ov::test;
using namespace ov::test::utils;

namespace {

using LSTMSubgraphParams =
        std::tuple<std::vector<ov::test::InputShape>, ov::element::Type, ov::op::RecurrentSequenceDirection>;
using OutputVector = std::vector<ov::Output<ov::Node>>;

class LSTMSubgraphNPUTest : public testing::WithParamInterface<LSTMSubgraphParams>, public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LSTMSubgraphParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << "IS=";

        ov::element::Type inputType;
        std::vector<ov::test::InputShape> shapes;
        ov::op::RecurrentSequenceDirection direction;

        std::tie(shapes, inputType, direction) = obj.param;

        for (auto shape : shapes) {
            result << vec2str(shape.second) << sep;
        }
        result << "direction=" << direction << sep;

        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        const int32_t startFrom = 0;
        const int32_t range = 100;

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInputs[i].get_element_type(),
                                                                        targetInputStaticShapes[i], range, startFrom);
            inputs.insert({funcInputs[i].get_node_shared_ptr(), tensor});
        }
    }

protected:
    //                    *------------------------*
    //                    |     Input parameter    |
    //                    |       (dynamic)        |
    //                    *------------------------*
    //                                 |
    //                  *----------------------------*
    //      ____________|           Reshape          |___
    //     |            *----------------------------*   |
    //     |                                             |
    //     |    *---------*  *---------*      *------------------*
    //     |    |  Const  |  |  Const  |      |     ShapeOf      |
    //     |    *---------*  *---------*      *------------------*
    //     |         |            |                      |
    //     |         |            |           *------------------*
    //     |         |            |           |      Gather      |
    //     |         |            |           *------------------*
    //     |         |            |                      |
    //     |         |            |                      |
    //     |         |            |                      |
    //     |         |_________*-----------------------------------*
    //     |___________________|    TensorIterator (body: LSTM)    |
    //                         *-----------------------------------*
    void SetUp() override {
        const auto& [shapes, typeForInput, direction] = this->GetParam();
        const auto& inputShapeForParameter = shapes[0];

        size_t hidden_size = 128;
        size_t input_size = 64;
        size_t batch_size = 1;

        size_t num_directions = direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;

        const auto dataShape = ov::test::InputShape{{batch_size, num_directions, hidden_size},
                                                    {{batch_size, num_directions, hidden_size}}};

        std::vector<ov::test::InputShape> inShapes = {inputShapeForParameter, dataShape, dataShape};

        init_input_shapes(inShapes);
        ov::ParameterVector inputParams;

        inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(typeForInput, inputDynamicShapes[0]));

        std::vector<int64_t> targetShape{1, -1, 64};
        auto reshapeConst =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{targetShape.size()}, targetShape);

        inputParams[0]->set_friendly_name("input_0");
        auto reshapedInput = std::make_shared<ov::opset1::Reshape>(inputParams[0], reshapeConst, true);

        auto X = reshapedInput->output(0);
        auto Y = std::make_shared<ov::op::v0::Parameter>(typeForInput,
                                                         ov::Shape{batch_size, num_directions, hidden_size});
        auto Z = std::make_shared<ov::op::v0::Parameter>(typeForInput,
                                                         ov::Shape{batch_size, num_directions, hidden_size});

        auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(X);
        auto indices = ov::op::v0::Constant::create(ov::element::i32, {1}, {1});
        auto axis = ov::op::v0::Constant::create(ov::element::i32, {}, {0});
        auto seq_lengths = std::make_shared<ov::op::v1::Gather>(shape_of, indices, axis);

        auto w_val = std::vector<float>(num_directions * 4 * hidden_size * input_size, 0);
        auto r_val = std::vector<float>(num_directions * 4 * hidden_size * hidden_size, 0);
        auto b_val = std::vector<float>(num_directions * 4 * hidden_size, 0);

        auto W = ov::op::v0::Constant::create(typeForInput, ov::Shape{num_directions, 4 * hidden_size, input_size},
                                              w_val);
        auto R = ov::op::v0::Constant::create(typeForInput, ov::Shape{num_directions, 4 * hidden_size, hidden_size},
                                              r_val);
        auto B = ov::op::v0::Constant::create(typeForInput, ov::Shape{num_directions, 4 * hidden_size}, b_val);

        auto lstm_sequence =
                std::make_shared<ov::op::v5::LSTMSequence>(X, Y, Z, seq_lengths, W, R, B, hidden_size, direction);

        auto Y_out = std::make_shared<ov::op::v0::Result>(lstm_sequence->output(0));
        auto Ho = std::make_shared<ov::op::v0::Result>(lstm_sequence->output(1));
        auto Co = std::make_shared<ov::op::v0::Result>(lstm_sequence->output(2));

        Y_out->set_friendly_name("Y_out");
        Ho->set_friendly_name("Ho");
        Co->set_friendly_name("Co");

        function =
                std::make_shared<ov::Model>(ov::NodeVector{Y_out, Ho, Co}, ov::ParameterVector{inputParams[0], Y, Z});
        function->set_friendly_name("LSTMSequenceSubgraphNPU");

        ov::pass::Manager manager;
        manager.register_pass<ov::pass::ConvertLSTMSequenceToTensorIterator>();
        manager.run_passes(function);
    }
};

TEST_P(LSTMSubgraphNPUTest, NPU4000_HW_TestKindSubgraph) {
    setMLIRCompilerType();
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

const std::vector<ov::element::Type> inputType = {ov::element::f32};
const std::vector<ov::op::RecurrentSequenceDirection> direction = {ov::op::RecurrentSequenceDirection::FORWARD,
                                                                   ov::op::RecurrentSequenceDirection::REVERSE,
                                                                   ov::op::RecurrentSequenceDirection::BIDIRECTIONAL};

const std::vector<std::vector<ov::test::InputShape>> inShapesShapeOfDataDynamic = {
        {{{ov::Dimension(1, 5), 1, 64}, {{4, 1, 64}}}}};

INSTANTIATE_TEST_SUITE_P(smoke_LSTMSubgraphNPUTest, LSTMSubgraphNPUTest,
                         ::testing::Combine(::testing::ValuesIn(inShapesShapeOfDataDynamic),
                                            ::testing::ValuesIn(inputType), ::testing::ValuesIn(direction)),
                         LSTMSubgraphNPUTest::getTestCaseName);

}  // namespace
