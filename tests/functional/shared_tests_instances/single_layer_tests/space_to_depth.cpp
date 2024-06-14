//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_op_tests/space_to_depth.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {

namespace test {

class SpaceToDepthLayerTestCommon : public SpaceToDepthLayerTest, virtual public VpuOv2LayerTest {};

class SpaceToDepthLayerTest_NPU3700 : public SpaceToDepthLayerTestCommon {};
class SpaceToDepthLayerTest_NPU3720 : public SpaceToDepthLayerTestCommon {};
class SpaceToDepthLayerTest_NPU4000 : public SpaceToDepthLayerTestCommon {};

TEST_P(SpaceToDepthLayerTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

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

using ov::test::SpaceToDepthLayerTest_NPU3700;
using ov::test::SpaceToDepthLayerTest_NPU3720;
using ov::test::SpaceToDepthLayerTest_NPU4000;

namespace {
const std::vector<ov::element::Type> inputTypes = {ov::element::f32,
                                                   ov::element::f16,  // value from CPU-plugin I16 is changed for FP16
                                                   ov::element::u8};

const std::vector<ov::op::v0::SpaceToDepth::SpaceToDepthMode> modes = {
        ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST,
        ov::op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST};

/* ============= NPU 3700 ============= */

const std::vector<std::vector<ov::Shape>> inputShapesBS2 = {
        {{1, 1, 2, 2}}, {{1, 1, 4, 4}}, {{1, 1, 6, 6}}, {{2, 8, 6, 6}}, {{2, 4, 10, 8}}};

const std::vector<std::vector<ov::Shape>> inputShapesBS3 = {
        {{1, 1, 3, 3}}, {{1, 1, 6, 6}}, {{1, 1, 9, 9}}, {{2, 4, 9, 9}}, {{2, 3, 15, 12}}};

const auto SpaceToDepthBS2 =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesBS2)),
                           ::testing::ValuesIn(inputTypes), ::testing::ValuesIn(modes), ::testing::Values(2),
                           ::testing::Values(ov::test::utils::DEVICE_NPU));

const auto SpaceToDepthBS3 =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesBS3)),
                           ::testing::ValuesIn(inputTypes), ::testing::ValuesIn(modes), ::testing::Values(3),
                           ::testing::Values(ov::test::utils::DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_SpaceToDepthBS2, SpaceToDepthLayerTest_NPU3700, SpaceToDepthBS2,
                         SpaceToDepthLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SpaceToDepthBS3, SpaceToDepthLayerTest_NPU3700, SpaceToDepthBS3,
                         SpaceToDepthLayerTest_NPU3700::getTestCaseName);

/* ============= NPU 3720/4000 ============= */

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
