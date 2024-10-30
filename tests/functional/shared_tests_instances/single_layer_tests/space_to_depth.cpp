//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <common_test_utils/ov_tensor_utils.hpp>
#include "common_test_utils/test_constants.hpp"
#include "single_op_tests/space_to_depth.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
namespace ov {

namespace test {

class SpaceToDepthLayerTestCommon : public SpaceToDepthLayerTest, virtual public VpuOv2LayerTest {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 8, 0, 32);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }
};

class SpaceToDepthLayerTest_NPU3720 : public SpaceToDepthLayerTestCommon {};
class SpaceToDepthLayerTest_NPU4000 : public SpaceToDepthLayerTestCommon {};

TEST_P(SpaceToDepthLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(SpaceToDepthLayerTest_NPU4000, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(SpaceToDepthLayerTest_NPU4000, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test

}  // namespace ov

using ov::test::SpaceToDepthLayerTest_NPU3720;
using ov::test::SpaceToDepthLayerTest_NPU4000;

namespace {
const std::vector<ov::element::Type> inputTypes = {ov::element::f32,
                                                   ov::element::f16,  // value from CPU-plugin I16 is changed for FP16
                                                   ov::element::u8};

const std::vector<ov::op::v0::SpaceToDepth::SpaceToDepthMode> modes = {
        ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST,
        ov::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST};

/* ============= NPU 3720/4000/5010 ============= */

const auto SpaceToDepthBS2_PRECOMMIT =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                   std::vector<std::vector<ov::Shape>>({{{1, 2, 3 * 4, 3 * 4}}}))),
                           ::testing::ValuesIn(inputTypes), ::testing::ValuesIn(modes), ::testing::Values(2),
                           ::testing::Values(ov::test::utils::DEVICE_NPU));

const auto SpaceToDepthBS3_PRECOMMIT =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                   std::vector<std::vector<ov::Shape>>({{{1, 2, 3 * 3, 3 * 3}}}))),
                           ::testing::ValuesIn(inputTypes), ::testing::ValuesIn(modes), ::testing::Values(3),
                           ::testing::Values(ov::test::utils::DEVICE_NPU));

const auto smoke_SpaceToDepthBS4_with_tiling =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                   std::vector<std::vector<ov::Shape>>({{{1, 48, 160, 80}}}))),
                           ::testing::ValuesIn(inputTypes), ::testing::ValuesIn(modes), ::testing::Values(4),
                           ::testing::Values(ov::test::utils::DEVICE_NPU));

/* ============= NPU 3720 ============= */

INSTANTIATE_TEST_SUITE_P(smoke_precommit_SpaceToDepthBS2, SpaceToDepthLayerTest_NPU3720, SpaceToDepthBS2_PRECOMMIT,
                         SpaceToDepthLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_SpaceToDepthBS3, SpaceToDepthLayerTest_NPU3720, SpaceToDepthBS3_PRECOMMIT,
                         SpaceToDepthLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_SpaceToDepth_with_tiling, SpaceToDepthLayerTest_NPU3720,
                         smoke_SpaceToDepthBS4_with_tiling, SpaceToDepthLayerTest_NPU3720::getTestCaseName);

/* ============= NPU 4000 ============= */

INSTANTIATE_TEST_SUITE_P(smoke_precommit_SpaceToDepthBS2, SpaceToDepthLayerTest_NPU4000, SpaceToDepthBS2_PRECOMMIT,
                         SpaceToDepthLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_SpaceToDepthBS3, SpaceToDepthLayerTest_NPU4000, SpaceToDepthBS3_PRECOMMIT,
                         SpaceToDepthLayerTest_NPU4000::getTestCaseName);

}  // namespace
