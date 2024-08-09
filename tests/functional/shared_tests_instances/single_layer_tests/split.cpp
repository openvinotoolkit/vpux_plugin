//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/split.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
namespace ov {

namespace test {

class SplitLayerTestCommon : public SplitLayerTest, virtual public VpuOv2LayerTest {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 8, 0, 32);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }
};

class SplitLayerTest_NPU3720 : public SplitLayerTestCommon {};
class SplitLayerTest_NPU4000 : public SplitLayerTestCommon {};

TEST_P(SplitLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(SplitLayerTest_NPU4000, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace test

}  // namespace ov

using ov::test::SplitLayerTest_NPU3720;
using ov::test::SplitLayerTest_NPU4000;

namespace {
const std::vector<ov::element::Type> modelTypes = {
        ov::element::f32,  // Testing FP32/FP16 netPrecision functionality only for small scope of
        ov::element::f16   // tests: GRNLayerTest, SplitLayerTest, CTCGreedyDecoderLayerTest
};

const auto paramsConfig1 =
        testing::Combine(::testing::Values(2, 3), ::testing::Values(0, 1, 2, 3), ::testing::ValuesIn(modelTypes),
                         ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                 std::vector<std::vector<ov::Shape>>({{{6, 6, 12, 24}}}))),
                         ::testing::Values(std::vector<size_t>({})), ::testing::Values(ov::test::utils::DEVICE_NPU));
const auto paramsConfig2 =
        testing::Combine(::testing::Values(2, 3), ::testing::Values(1, 2, 3), ::testing::ValuesIn(modelTypes),
                         ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                 std::vector<std::vector<ov::Shape>>({{{6, 6, 12, 24}}}))),
                         ::testing::Values(std::vector<size_t>({})), ::testing::Values(ov::test::utils::DEVICE_NPU));
const auto paramsPrecommit =
        testing::Combine(::testing::Values(2, 3), ::testing::Values(0), ::testing::ValuesIn(modelTypes),
                         ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                 std::vector<std::vector<ov::Shape>>({{{6, 6, 12, 24}}}))),
                         ::testing::Values(std::vector<size_t>({})), ::testing::Values(ov::test::utils::DEVICE_NPU));

// --------- NPU3720 ---------

INSTANTIATE_TEST_SUITE_P(smoke_Split, SplitLayerTest_NPU3720, paramsConfig1, SplitLayerTest_NPU3720::getTestCaseName);

// --------- NPU4000 ---------

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Split, SplitLayerTest_NPU4000, paramsPrecommit,
                         SplitLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Split, SplitLayerTest_NPU4000, paramsConfig2, SplitLayerTest_NPU4000::getTestCaseName);

}  // namespace
