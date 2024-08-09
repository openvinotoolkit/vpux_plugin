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

using ov::test::LSTMSequenceLayerTest_NPU3720;
using ov::test::LSTMSequenceLayerTest_NPU4000;

namespace {
std::vector<ov::test::utils::SequenceTestsMode> mode = {
        ov::test::utils::SequenceTestsMode::PURE_SEQ,
};

std::vector<std::vector<std::string>> activations = {{"sigmoid", "tanh", "tanh"}};
std::vector<float> clip{0.f};
std::vector<ov::op::RecurrentSequenceDirection> direction = {ov::op::RecurrentSequenceDirection::FORWARD,
                                                             ov::op::RecurrentSequenceDirection::REVERSE,
                                                             ov::op::RecurrentSequenceDirection::BIDIRECTIONAL};
std::vector<ov::element::Type> modelTypes = {ov::element::f16};

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
