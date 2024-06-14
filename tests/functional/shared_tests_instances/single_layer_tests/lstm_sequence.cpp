// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/lstm_sequence.hpp"
#include <vector>
#include "openvino/op/util/attr_types.hpp"
#include "shared_tests_instances/vpu_ov2_layer_test.hpp"

#include <memory>
#include <string>
#include <tuple>
#include "common_test_utils/node_builders/lstm_cell.hpp"

namespace ov {

namespace test {

class LSTMSequenceLayerTest_NPU3700 : public LSTMSequenceTest, virtual public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LSTMSequenceParams>& obj) {
        ov::test::utils::SequenceTestsMode mode;
        size_t seq_lengths;
        size_t batch;
        size_t hidden_size;
        size_t input_size;
        std::vector<std::string> activations;
        std::vector<float> activations_alpha;
        std::vector<float> activations_beta;
        float clip;
        ov::op::RecurrentSequenceDirection direction;
        ov::test::utils::InputLayerType WRBType;
        ov::element::Type modelType;
        std::string targetDevice;
        std::tie(mode, seq_lengths, batch, hidden_size, input_size, activations, clip, direction, WRBType, modelType,
                 targetDevice) = obj.param;
        std::vector<ov::Shape> inputShapes = {
                {{batch, input_size},
                 {batch, hidden_size},
                 {batch, hidden_size},
                 {4 * hidden_size, input_size},
                 {4 * hidden_size, hidden_size},
                 {4 * hidden_size}},
        };
        std::ostringstream result;
        result << "mode=" << mode << "_";
        result << "seq_lengths=" << seq_lengths << "_";
        result << "batch=" << batch << "_";
        result << "hidden_size=" << hidden_size << "_";
        result << "input_size=" << input_size << "_";
        result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
        result << "activations=" << ov::test::utils::vec2str(activations) << "_";
        result << "direction=" << direction << "_";
        result << "clip=" << clip << "_";
        result << "WRBType=" << WRBType << "_";
        result << "netPRC=" << modelType.get_type_name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        return result.str();
    }

protected:
    void SetUp() override {
        size_t seq_lengths;

        size_t batch;
        size_t hidden_size;
        size_t input_size;
        std::vector<std::string> activations;
        std::vector<float> activations_alpha;
        std::vector<float> activations_beta;
        float clip;
        ov::op::RecurrentSequenceDirection direction;
        ov::test::utils::InputLayerType WRBType;
        ov::element::Type modelType;
        std::tie(m_mode, seq_lengths, batch, hidden_size, input_size, activations, clip, direction, WRBType, modelType,
                 targetDevice) = this->GetParam();

        size_t num_directions = direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;
        std::vector<ov::Shape> inputShapes = {
                {{batch, seq_lengths, input_size},
                 {batch, num_directions, hidden_size},
                 {batch, num_directions, hidden_size},
                 {batch},
                 {num_directions, 4 * hidden_size, input_size},
                 {num_directions, 4 * hidden_size, hidden_size},
                 {num_directions, 4 * hidden_size}},
        };
        init_input_shapes(ov::test::static_shapes_to_test_representation(inputShapes));
        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(modelType, targetStaticShapes.front().at(0)),
                std::make_shared<ov::op::v0::Parameter>(modelType, targetStaticShapes.front().at(1)),
                std::make_shared<ov::op::v0::Parameter>(modelType, targetStaticShapes.front().at(2))};

        ov::OutputVector param_outs;
        for (const auto& param : params) {
            param_outs.push_back(param);
        }

        ASSERT_EQ(ov::test::utils::InputLayerType::CONSTANT, WRBType);
        std::vector<ov::Shape> WRB = {targetStaticShapes.front().at(4), targetStaticShapes.front().at(5),
                                      targetStaticShapes.front().at(6), targetStaticShapes.front().at(3)};
        auto lstm_sequence = ov::test::utils::make_lstm(param_outs, WRB, hidden_size, activations, {}, {}, clip, true,
                                                        direction, m_mode);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(lstm_sequence->output(0)),
                                 std::make_shared<ov::op::v0::Result>(lstm_sequence->output(1)),
                                 std::make_shared<ov::op::v0::Result>(lstm_sequence->output(2))};
        function = std::make_shared<ov::Model>(results, params, "lstm_sequence");
    }

private:
    ov::test::utils::SequenceTestsMode m_mode;
    int64_t m_max_seq_len = 0;
};

class LSTMSequenceLayerTest_NPU3720 : public LSTMSequenceTest, virtual public VpuOv2LayerTest {
    void SetUp() override {
        inType = ov::element::f16;
        outType = ov::element::f16;
        LSTMSequenceTest::SetUp();
    }
};

class LSTMSequenceLayerTest_NPU4000 : public LSTMSequenceTest, virtual public VpuOv2LayerTest {
    void SetUp() override {
        inType = ov::element::f16;
        outType = ov::element::f16;
        LSTMSequenceTest::SetUp();
    }
};

TEST_P(LSTMSequenceLayerTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

TEST_P(LSTMSequenceLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(LSTMSequenceLayerTest_NPU4000, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
}  // namespace test

}  // namespace ov

using ov::test::LSTMSequenceLayerTest_NPU3700;
using ov::test::LSTMSequenceLayerTest_NPU3720;
using ov::test::LSTMSequenceLayerTest_NPU4000;

namespace {
std::vector<ov::test::utils::SequenceTestsMode> mode = {
        ov::test::utils::SequenceTestsMode::PURE_SEQ,
};

// --------- NPU3700 ---------
std::vector<size_t> seq_lengths_zero_clip3700{1};
std::vector<size_t> batch3700{1};
std::vector<size_t> hidden_size3700{1};
std::vector<size_t> input_size3700{1};
std::vector<std::vector<std::string>> activations = {{"sigmoid", "tanh", "tanh"}};
std::vector<float> clip{0.f};
std::vector<ov::op::RecurrentSequenceDirection> direction = {ov::op::RecurrentSequenceDirection::FORWARD,
                                                             ov::op::RecurrentSequenceDirection::REVERSE,
                                                             ov::op::RecurrentSequenceDirection::BIDIRECTIONAL};
std::vector<ov::element::Type> modelTypes = {ov::element::f16};

INSTANTIATE_TEST_CASE_P(smoke_LSTMSequenceCommonZeroClip, LSTMSequenceLayerTest_NPU3700,
                        ::testing::Combine(::testing::ValuesIn(mode), ::testing::ValuesIn(seq_lengths_zero_clip3700),
                                           ::testing::ValuesIn(batch3700), ::testing::ValuesIn(hidden_size3700),
                                           ::testing::ValuesIn(input_size3700), ::testing::ValuesIn(activations),
                                           ::testing::ValuesIn(clip), ::testing::ValuesIn(direction),
                                           ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                           ::testing::ValuesIn(modelTypes),
                                           ::testing::Values(ov::test::utils::DEVICE_NPU)),
                        LSTMSequenceLayerTest_NPU3700::getTestCaseName);

// --------- NPU3720/4000 ---------
std::vector<size_t> seq_lengths_zero_clip{3};
std::vector<size_t> batch{3};
std::vector<size_t> hidden_size{64};
std::vector<size_t> input_size{67};

const auto lstmConfig =
        ::testing::Combine(::testing::ValuesIn(mode), ::testing::ValuesIn(seq_lengths_zero_clip),
                           ::testing::ValuesIn(batch), ::testing::ValuesIn(hidden_size),
                           ::testing::ValuesIn(input_size), ::testing::ValuesIn(activations), ::testing::ValuesIn(clip),
                           ::testing::ValuesIn(direction), ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                           ::testing::ValuesIn(modelTypes), ::testing::Values(ov::test::utils::DEVICE_NPU));

INSTANTIATE_TEST_CASE_P(smoke_precommit_LSTMSequenceCommonZeroClip, LSTMSequenceLayerTest_NPU3720, lstmConfig,
                        LSTMSequenceLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_LSTMSequenceCommonZeroClip, LSTMSequenceLayerTest_NPU4000, lstmConfig,
                        LSTMSequenceLayerTest_NPU4000::getTestCaseName);

}  // namespace
