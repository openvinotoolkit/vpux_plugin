//
// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/concat.hpp"
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov2_layer_test.hpp"

#include <vector>

using namespace ov::test::utils;

namespace ov {
namespace test {

class ConcatLayerTestCommon : public ConcatLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(ConcatLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ConcatLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

std::vector<int> axes = {0, 1, 2, 3};

std::vector<std::vector<ov::Shape>> inShapes = {
        {{10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
        {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}}};

std::vector<ov::element::Type> netPrecisions = {ov::element::f16, ov::element::u8};
// Check parameters from InceptionV3
// This test is just attempt to use parameters other than in CPU-plugin.
// Note: NPU-plugin does not support batch-size > 1.
std::vector<int> axes_check = {1};

std::vector<std::vector<ov::Shape>> inShapes_check = {
        {{1, 64, 35, 35}, {1, 64, 35, 35}}, {{1, 64, 35, 35}, {1, 64, 35, 35}, {1, 96, 35, 35}, {1, 32, 35, 35}}};

const auto concatParams = ::testing::Combine(
        ::testing::ValuesIn(axes),
        ::testing::Values(ov::test::static_shapes_to_test_representation({{1, 16, 10, 10}, {1, 16, 10, 10}})),
        ::testing::Values(ov::element::u8), ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Concat, ConcatLayerTestCommon, concatParams,
                         ConcatLayerTestCommon::getTestCaseName);

}  // namespace
