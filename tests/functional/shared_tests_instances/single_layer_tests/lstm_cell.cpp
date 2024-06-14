// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_op_tests/lstm_cell.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class LSTMCellLayerTest_NPU3700 : public LSTMCellTest, virtual public VpuOv2LayerTest {};

TEST_P(LSTMCellLayerTest_NPU3700, HW) {
    rel_threshold = 0.06;
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

class LSTMCellLayerTestCommon : public LSTMCellTest, virtual public VpuOv2LayerTest {};

class LSTMCellLayerTest_NPU3720 : public LSTMCellLayerTestCommon {};
class LSTMCellLayerTest_NPU4000 : public LSTMCellLayerTestCommon {};

TEST_P(LSTMCellLayerTest_NPU3720, HW) {
    rel_threshold = 0.06;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(LSTMCellLayerTest_NPU4000, HW) {
    rel_threshold = 0.06;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {
std::vector<bool> should_decompose{false};
std::vector<std::vector<std::string>> activations = {{"sigmoid", "tanh", "tanh"}};
std::vector<float> clip{0.f};
std::vector<ov::element::Type> modelTypes = {ov::element::f16};

// NPU3700 tests
std::vector<size_t> batch3700{1};
std::vector<size_t> hidden_size3700{1};
std::vector<size_t> input_size3700{5};

// Test scope was reduced to one test due to accuracy drop in many cases (accuracy deviation ~0.2(5%) from reference)
// that can't be moved to separate scope due to dissimilar parameters
// Also some simple test cases with small dimensions takes more than 5 sec time (don't fit to timeout)

INSTANTIATE_TEST_CASE_P(smoke_LSTMCellCommon, LSTMCellLayerTest_NPU3700,
                        ::testing::Combine(::testing::ValuesIn(should_decompose), ::testing::ValuesIn(batch3700),
                                           ::testing::ValuesIn(hidden_size3700), ::testing::ValuesIn(input_size3700),
                                           ::testing::ValuesIn(activations), ::testing::ValuesIn(clip),
                                           ::testing::Values(InputLayerType::CONSTANT),
                                           ::testing::Values(InputLayerType::CONSTANT),
                                           ::testing::Values(InputLayerType::CONSTANT), ::testing::ValuesIn(modelTypes),
                                           ::testing::Values(DEVICE_NPU)),
                        LSTMCellLayerTest_NPU3700::getTestCaseName);

// NPU3720/4000 tests
std::vector<size_t> batch{1};
std::vector<size_t> hidden_size{4, 64};
std::vector<size_t> input_size{6, 24};
std::vector<size_t> hidden_size_precommit{2, 16};
std::vector<size_t> input_size_precommit{3, 12};

const auto lstmCellConfig = ::testing::Combine(
        ::testing::ValuesIn(should_decompose), ::testing::ValuesIn(batch), ::testing::ValuesIn(hidden_size),
        ::testing::ValuesIn(input_size), ::testing::ValuesIn(activations), ::testing::ValuesIn(clip),
        ::testing::Values(InputLayerType::CONSTANT), ::testing::Values(InputLayerType::CONSTANT),
        ::testing::Values(InputLayerType::CONSTANT), ::testing::ValuesIn(modelTypes), ::testing::Values(DEVICE_NPU));

const auto lstmCellPrecommitConfig = ::testing::Combine(
        ::testing::ValuesIn(should_decompose), ::testing::ValuesIn(batch), ::testing::ValuesIn(hidden_size_precommit),
        ::testing::ValuesIn(input_size_precommit), ::testing::ValuesIn(activations), ::testing::ValuesIn(clip),
        ::testing::Values(InputLayerType::CONSTANT), ::testing::Values(InputLayerType::CONSTANT),
        ::testing::Values(InputLayerType::CONSTANT), ::testing::ValuesIn(modelTypes), ::testing::Values(DEVICE_NPU));

// ------ NPU3720 ------

INSTANTIATE_TEST_CASE_P(smoke_precommit_LSTMCellCommon, LSTMCellLayerTest_NPU3720, lstmCellPrecommitConfig,
                        LSTMCellTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_LSTMCellCommon, LSTMCellLayerTest_NPU3720, lstmCellConfig, LSTMCellTest::getTestCaseName);

// ------ NPU4000 ------

INSTANTIATE_TEST_CASE_P(smoke_precommit_LSTMCellCommon, LSTMCellLayerTest_NPU4000, lstmCellPrecommitConfig,
                        LSTMCellTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_LSTMCellCommon, LSTMCellLayerTest_NPU4000, lstmCellConfig, LSTMCellTest::getTestCaseName);

}  // namespace
