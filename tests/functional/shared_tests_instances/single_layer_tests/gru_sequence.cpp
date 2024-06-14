//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_op_tests/gru_sequence.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include <vector>
#include "openvino/pass/manager.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class GRUSequenceLayerTestCommon : public GRUSequenceTest, virtual public VpuOv2LayerTest {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        SubgraphBaseTest::generate_inputs(targetInputStaticShapes);
        ov::Tensor blob;
        const int64_t maxSeqLen = targetInputStaticShapes[0][1];
        const auto& funcInputs = function->inputs();
        if (funcInputs.size() > 2) {
            const auto& seqLenInput = inputs.find(funcInputs[2].get_node_shared_ptr());
            if (seqLenInput == inputs.end())
                blob = create_and_fill_tensor_normal_distribution(funcInputs[0].get_element_type(),
                                                                  targetInputStaticShapes[0], 0, 1, 1);

            blob = create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], maxSeqLen, 0,
                                          1);
            inputs.insert({funcInputs[0].get_node_shared_ptr(), blob});
        }
    }
};

class GRUSequenceLayerTest_NPU3720 : public GRUSequenceLayerTestCommon {};
class GRUSequenceLayerTest_NPU4000 : public GRUSequenceLayerTestCommon {};

TEST_P(GRUSequenceLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(GRUSequenceLayerTest_NPU4000, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;
using GRUDirection = ov::op::RecurrentSequenceDirection;

namespace {

const auto testMode = SequenceTestsMode::PURE_SEQ;
const std::vector<std::string> activations = {"sigmoid", "tanh"};
const float clip = 0.0f;
const std::vector<bool> shouldLinearBeforeReset{true, false};
const std::vector<GRUDirection> directionMode{GRUDirection::FORWARD, GRUDirection::REVERSE};

const std::vector<GRUDirection> directionModeBi{GRUDirection::BIDIRECTIONAL};

const ov::element::Type modelTypes = ov::element::f16;

const std::vector<std::vector<ov::Shape>> iShape = {
        // {batch, seq_lengths, input_size}, {batch, num_direction, hidden_size}, {batch},
        {{2, 5, 10}, {2, 1, 4}, {2}},
};
const std::vector<std::vector<ov::Shape>> iShapeTiling = {
        {{100, 1000, 10}, {100, 1, 1}, {100}},
};

const std::vector<std::vector<ov::Shape>> iShapeSplit = {
        {{1, 1, 10}, {1, 1, 569}, {1}},
};

const std::vector<std::vector<ov::Shape>> iShapeSplit1 = {
        {{1, 1, 10}, {1, 1, 200}, {1}},
};

// Direction mode = BIDIRECTIONAL
const std::vector<std::vector<ov::Shape>> iShapeBi = {
        // {batch, seq_lengths, input_size}, {batch, num_direction, hidden_size}, {batch},
        {{2, 5, 10}, {2, 2, 4}, {2}},
};
const std::vector<std::vector<ov::Shape>> iShapeTilingBi = {
        {{100, 1000, 10}, {100, 2, 1}, {100}},
};

const std::vector<std::vector<ov::Shape>> iShapeSplitBi = {
        {{1, 1, 10}, {1, 2, 569}, {1}},
};

const std::vector<std::vector<ov::Shape>> iShapeSplit1Bi = {
        {{1, 1, 10}, {1, 2, 200}, {1}},
};

const auto gruSequenceParam0 = testing::Combine(
        ::testing::Values(testMode), ::testing::ValuesIn(static_shapes_to_test_representation(iShape)),
        ::testing::Values(activations), ::testing::Values(clip), ::testing::ValuesIn(shouldLinearBeforeReset),
        ::testing::ValuesIn(directionMode), ::testing::Values(InputLayerType::CONSTANT), ::testing::Values(modelTypes),
        ::testing::Values(DEVICE_NPU));

const auto gruSequenceParam0Bi = testing::Combine(
        ::testing::Values(testMode), ::testing::ValuesIn(static_shapes_to_test_representation(iShapeBi)),
        ::testing::Values(activations), ::testing::Values(clip), ::testing::ValuesIn(shouldLinearBeforeReset),
        ::testing::ValuesIn(directionModeBi), ::testing::Values(InputLayerType::CONSTANT),
        ::testing::Values(modelTypes), ::testing::Values(DEVICE_NPU));

const auto gruSequenceParam1 = testing::Combine(
        ::testing::Values(testMode), ::testing::ValuesIn(static_shapes_to_test_representation(iShapeTiling)),
        ::testing::Values(activations), ::testing::Values(clip), ::testing::ValuesIn(shouldLinearBeforeReset),
        ::testing::ValuesIn(directionMode), ::testing::Values(InputLayerType::CONSTANT), ::testing::Values(modelTypes),
        ::testing::Values(DEVICE_NPU));

const auto gruSequenceParam1Bi = testing::Combine(
        ::testing::Values(testMode), ::testing::ValuesIn(static_shapes_to_test_representation(iShapeTilingBi)),
        ::testing::Values(activations), ::testing::Values(clip), ::testing::ValuesIn(shouldLinearBeforeReset),
        ::testing::ValuesIn(directionModeBi), ::testing::Values(InputLayerType::CONSTANT),
        ::testing::Values(modelTypes), ::testing::Values(DEVICE_NPU));

const auto gruSequenceParam2 = testing::Combine(
        ::testing::Values(testMode), ::testing::ValuesIn(static_shapes_to_test_representation(iShapeSplit)),
        ::testing::Values(activations), ::testing::Values(clip), ::testing::ValuesIn(shouldLinearBeforeReset),
        ::testing::ValuesIn(directionMode), ::testing::Values(InputLayerType::CONSTANT), ::testing::Values(modelTypes),
        ::testing::Values(DEVICE_NPU));

const auto gruSequenceParam2Bi = testing::Combine(
        ::testing::Values(testMode), ::testing::ValuesIn(static_shapes_to_test_representation(iShapeSplitBi)),
        ::testing::Values(activations), ::testing::Values(clip), ::testing::ValuesIn(shouldLinearBeforeReset),
        ::testing::ValuesIn(directionModeBi), ::testing::Values(InputLayerType::CONSTANT),
        ::testing::Values(modelTypes), ::testing::Values(DEVICE_NPU));

const auto gruSequenceParam3 = testing::Combine(
        ::testing::Values(testMode), ::testing::ValuesIn(static_shapes_to_test_representation(iShapeSplit1)),
        ::testing::Values(activations), ::testing::Values(clip), ::testing::ValuesIn(shouldLinearBeforeReset),
        ::testing::ValuesIn(directionMode), ::testing::Values(InputLayerType::CONSTANT), ::testing::Values(modelTypes),
        ::testing::Values(DEVICE_NPU));

const auto gruSequenceParam3Bi = testing::Combine(
        ::testing::Values(testMode), ::testing::ValuesIn(static_shapes_to_test_representation(iShapeSplit1Bi)),
        ::testing::Values(activations), ::testing::Values(clip), ::testing::ValuesIn(shouldLinearBeforeReset),
        ::testing::ValuesIn(directionModeBi), ::testing::Values(InputLayerType::CONSTANT),
        ::testing::Values(modelTypes), ::testing::Values(DEVICE_NPU));

//    NPU3720
INSTANTIATE_TEST_SUITE_P(smoke_precommit_GRUSequence, GRUSequenceLayerTest_NPU3720, gruSequenceParam0,
                         GRUSequenceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GRUSequence_Tiling, GRUSequenceLayerTest_NPU3720, gruSequenceParam1,
                         GRUSequenceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GRUSequence_Split, GRUSequenceLayerTest_NPU3720, gruSequenceParam2,
                         GRUSequenceTest::getTestCaseName);

// BIDIRECTIONAL NPU3720

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GRUSequence_BI, GRUSequenceLayerTest_NPU3720, gruSequenceParam0Bi,
                         GRUSequenceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GRUSequence_Tiling_BI, GRUSequenceLayerTest_NPU3720, gruSequenceParam1Bi,
                         GRUSequenceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GRUSequence_Split_BI, GRUSequenceLayerTest_NPU3720, gruSequenceParam2Bi,
                         GRUSequenceTest::getTestCaseName);

//    NPU4000
INSTANTIATE_TEST_SUITE_P(smoke_precommit_GRUSequence, GRUSequenceLayerTest_NPU4000, gruSequenceParam0,
                         GRUSequenceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GRUSequence_Tiling, GRUSequenceLayerTest_NPU4000, gruSequenceParam1,
                         GRUSequenceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GRUSequence_Split, GRUSequenceLayerTest_NPU4000, gruSequenceParam3,
                         GRUSequenceTest::getTestCaseName);
// BIDIRECTIONAL NPU4000

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GRUSequence_BI, GRUSequenceLayerTest_NPU4000, gruSequenceParam0Bi,
                         GRUSequenceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GRUSequence_Tiling_BI, GRUSequenceLayerTest_NPU4000, gruSequenceParam1Bi,
                         GRUSequenceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GRUSequence_Split_BI, GRUSequenceLayerTest_NPU4000, gruSequenceParam3Bi,
                         GRUSequenceTest::getTestCaseName);

}  // namespace
