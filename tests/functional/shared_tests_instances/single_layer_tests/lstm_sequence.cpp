// Copyright (C) 2018-2024 Intel Corporation
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

class LSTMSequenceLayerTestCommon : public LSTMSequenceTest, virtual public VpuOv2LayerTest {
    void SetUp() override {
        inType = ov::element::f16;
        outType = ov::element::f16;
        LSTMSequenceTest::SetUp();
    }
};

TEST_P(LSTMSequenceLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(LSTMSequenceLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;
namespace {
std::vector<utils::SequenceTestsMode> mode = {
        utils::SequenceTestsMode::PURE_SEQ,
};

std::vector<std::vector<std::string>> activations = {{"sigmoid", "tanh", "tanh"}};
std::vector<float> clip{0.f};
std::vector<ov::op::RecurrentSequenceDirection> direction = {ov::op::RecurrentSequenceDirection::FORWARD,
                                                             ov::op::RecurrentSequenceDirection::REVERSE,
                                                             ov::op::RecurrentSequenceDirection::BIDIRECTIONAL};
std::vector<ov::element::Type> modelTypes = {ov::element::f16};

std::vector<size_t> seq_lengths_zero_clip{3};
std::vector<size_t> batch{3};
std::vector<size_t> hidden_size{64};
std::vector<size_t> input_size{67};

const auto lstmConfig = ::testing::Combine(
        ::testing::ValuesIn(mode), ::testing::ValuesIn(seq_lengths_zero_clip), ::testing::ValuesIn(batch),
        ::testing::ValuesIn(hidden_size), ::testing::ValuesIn(input_size), ::testing::ValuesIn(activations),
        ::testing::ValuesIn(clip), ::testing::ValuesIn(direction), ::testing::Values(utils::InputLayerType::CONSTANT),
        ::testing::ValuesIn(modelTypes), ::testing::Values(utils::DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_LSTMSequenceCommonZeroClip, LSTMSequenceLayerTestCommon, lstmConfig,
                         LSTMSequenceLayerTestCommon::getTestCaseName);

// --------- NPU4000 Target speed up scenario---------
std::vector<size_t> seq_lengthsPt{2};  // 160 real case reduced for speed reason
std::vector<size_t> batchPt{1, 2};
std::vector<size_t> hidden_sizePt{16, 17, 64, 128, 144};
std::vector<size_t> input_sizePt{64};
std::vector<float> clipPt{0.f};
std::vector<ov::op::RecurrentSequenceDirection> directionPt = {ov::op::RecurrentSequenceDirection::BIDIRECTIONAL,
                                                               ov::op::RecurrentSequenceDirection::REVERSE};
// FORWARD BIDIRECTIONAL
const auto lstmConfigPt = ::testing::Combine(
        ::testing::ValuesIn(mode), ::testing::ValuesIn(seq_lengthsPt), ::testing::ValuesIn(batchPt),
        ::testing::ValuesIn(hidden_sizePt), ::testing::ValuesIn(input_sizePt), ::testing::ValuesIn(activations),
        ::testing::ValuesIn(clipPt), ::testing::ValuesIn(directionPt),
        ::testing::Values(ov::test::utils::InputLayerType::CONSTANT), ::testing::ValuesIn(modelTypes),
        ::testing::Values(ov::test::utils::DEVICE_NPU));

INSTANTIATE_TEST_CASE_P(smoke_precommit_LSTMSequencePt, LSTMSequenceLayerTestCommon, lstmConfigPt,
                        LSTMSequenceLayerTestCommon::getTestCaseName);

}  // namespace
