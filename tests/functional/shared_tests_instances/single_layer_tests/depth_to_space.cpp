//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/depth_to_space.hpp"
#include <openvino/opsets/opset3.hpp>
#include <vector>
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class DepthToSpaceLayerTestCommon : public DepthToSpaceLayerTest, virtual public VpuOv2LayerTest {};

class DepthToSpaceLayerTest_NPU3700 : public DepthToSpaceLayerTestCommon {};
class DepthToSpaceLayerTest_NPU3720 : public DepthToSpaceLayerTestCommon {};
class DepthToSpaceLayerTest_NPU4000 : public DepthToSpaceLayerTestCommon {};

TEST_P(DepthToSpaceLayerTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

TEST_P(DepthToSpaceLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DepthToSpaceLayerTest_NPU4000, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(DepthToSpaceLayerTest_NPU4000, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;
using ov::op::v0::DepthToSpace;

namespace {
const std::vector<ov::element::Type> inputPrecisions = {
        ov::element::f32, ov::element::u8,
        ov::element::f16,  // CPU-plugin has parameter I16, but NPU3700 does not
};                         // support it. So I16 is changed to FP16.

const std::vector<ov::op::v0::DepthToSpace::DepthToSpaceMode> modes = {
        ov::op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
        ov::op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST};

// ------ NPU3700 ------
const std::vector<std::vector<ov::Shape>> inputShapesBS2 = {
        {{1, 4, 1, 1}}, {{1, 4, 2, 2}}, {{1, 4, 3, 3}}, {{2, 32, 3, 3}}, {{2, 16, 5, 4}}};
// {1, 8, 1, 1, 1}, {1, 8, 2, 2, 2}, {1, 8, 3, 3, 3}, {2, 32, 3, 3, 3}, {2, 16, 5, 4, 6}};
// These 5-dimensional values from CPU-test, but NPU-plugin does not support dims.size() > 4.
// Therefore they are commented.
// For details please see: NPU-plugin/src/utils/dims_parser.cpp

const std::vector<std::vector<ov::Shape>> inputShapesBS3 = {
        {{1, 9, 1, 1}}, {{1, 9, 2, 2}}, {{1, 9, 3, 3}}, {{2, 36, 3, 3}}, {{2, 27, 5, 4}}};
// {1, 27, 1, 1, 1}, {1, 27, 2, 2, 2}, {1, 27, 3, 3, 3}, {2, 108, 3, 3, 3}, {2, 54, 5, 4, 6}};
// These 5-dimensional values from CPU-test, but NPU-plugin does not support dims.size() > 4.
// Therefore they are commented.
// For details please see: NPU-plugin/src/utils/dims_parser.cpp

const auto DepthToSpaceBS2 =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesBS2)),  // Input shape
                           ::testing::ValuesIn(inputPrecisions),                                       // Model type
                           ::testing::ValuesIn(modes),                                                 // Mode
                           ::testing::Values(2),                                                       // Block size
                           ::testing::Values(DEVICE_NPU));                                             // Device name

const auto DepthToSpaceBS3 = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesBS3)), ::testing::ValuesIn(inputPrecisions),
        ::testing::ValuesIn(modes), ::testing::Values(3), ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_DepthToSpaceBS2, DepthToSpaceLayerTest_NPU3700, DepthToSpaceBS2,
                         DepthToSpaceLayerTest_NPU3700::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DepthToSpaceBS3, DepthToSpaceLayerTest_NPU3700, DepthToSpaceBS3,
                         DepthToSpaceLayerTest_NPU3700::getTestCaseName);

// ------ NPU3720/4000 ------
const auto DepthToSpaceBS2_PRECOMMIT =
        ::testing::Combine(::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{1, 4, 3, 3}})}),
                           ::testing::ValuesIn(inputPrecisions), ::testing::ValuesIn(modes), ::testing::Values(2),
                           ::testing::Values(DEVICE_NPU));

const auto DepthToSpaceBS3_PRECOMMIT =
        ::testing::Combine(::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{1, 9, 3, 3}})}),
                           ::testing::ValuesIn(inputPrecisions), ::testing::ValuesIn(modes), ::testing::Values(3),
                           ::testing::Values(DEVICE_NPU));

const auto smoke_DepthToSpaceBS4_with_tiling =
        ::testing::Combine(::testing::ValuesIn({static_shapes_to_test_representation({ov::Shape{1, 48, 80, 80}})}),
                           ::testing::Values(ov::element::f16), ::testing::ValuesIn(modes), ::testing::Values(4),
                           ::testing::Values(DEVICE_NPU));

std::vector<std::vector<ov::Shape>> inputShapesBS4 = {
        {{1, 16, 5, 4}}, {{1, 16, 9, 7}}, {{1, 128, 5, 4}}, {{1, 128, 9, 7}}};
const auto smoke_DepthToSpaceBS4 = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesBS4)), ::testing::Values(ov::element::f16),
        ::testing::Values(DepthToSpace::DepthToSpaceMode::DEPTH_FIRST), ::testing::Values(4),
        ::testing::Values(DEVICE_NPU));

const auto DepthToSpaceBS5_with_large_height =
        ::testing::Combine(::testing::Values(static_shapes_to_test_representation({ov::Shape{1, 4, 300, 3}})),
                           ::testing::ValuesIn(inputPrecisions), ::testing::ValuesIn(modes), ::testing::Values(2),
                           ::testing::Values(DEVICE_NPU));

/* ============= NPU 3720 ============= */

INSTANTIATE_TEST_SUITE_P(smoke_precommit_DepthToSpaceBS2, DepthToSpaceLayerTest_NPU3720, DepthToSpaceBS2_PRECOMMIT,
                         DepthToSpaceLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_DepthToSpaceBS3, DepthToSpaceLayerTest_NPU3720, DepthToSpaceBS3_PRECOMMIT,
                         DepthToSpaceLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DepthToSpaceBS4, DepthToSpaceLayerTest_NPU3720, smoke_DepthToSpaceBS4,
                         DepthToSpaceLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DepthToSpace_with_tiling, DepthToSpaceLayerTest_NPU3720,
                         smoke_DepthToSpaceBS4_with_tiling, DepthToSpaceLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DepthToSpace_with_large_height, DepthToSpaceLayerTest_NPU3720,
                         DepthToSpaceBS5_with_large_height, DepthToSpaceLayerTest_NPU3720::getTestCaseName);

/* ============= NPU 4000 ============= */

INSTANTIATE_TEST_SUITE_P(smoke_precommit_DepthToSpaceBS2, DepthToSpaceLayerTest_NPU4000, DepthToSpaceBS2_PRECOMMIT,
                         DepthToSpaceLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_DepthToSpaceBS3, DepthToSpaceLayerTest_NPU4000, DepthToSpaceBS3_PRECOMMIT,
                         DepthToSpaceLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DepthToSpaceBS4, DepthToSpaceLayerTest_NPU4000, smoke_DepthToSpaceBS4,
                         DepthToSpaceLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DepthToSpace_with_tiling, DepthToSpaceLayerTest_NPU4000,
                         smoke_DepthToSpaceBS4_with_tiling, DepthToSpaceLayerTest_NPU4000::getTestCaseName);

}  // namespace
