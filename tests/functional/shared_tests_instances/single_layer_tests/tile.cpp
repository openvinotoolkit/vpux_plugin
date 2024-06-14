//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/tile.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {

namespace test {

class TileLayerTestCommon : public TileLayerTest, virtual public VpuOv2LayerTest {};

class TileLayerTest_NPU3700 : public TileLayerTestCommon {};
class TileLayerTest_NPU3720 : public TileLayerTestCommon {};
class TileLayerTest_NPU3720_tiling : public TileLayerTestCommon {};
class TileLayerTest_NPU4000 : public TileLayerTestCommon {};

TEST_P(TileLayerTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

TEST_P(TileLayerTest_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(TileLayerTest_NPU3720_tiling, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(TileLayerTest_NPU4000, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test

}  // namespace ov

using ov::test::TileLayerTest_NPU3700;
using ov::test::TileLayerTest_NPU3720;
using ov::test::TileLayerTest_NPU3720_tiling;
using ov::test::TileLayerTest_NPU4000;

namespace {

const std::vector<ov::element::Type> modelTypes = {ov::element::f16, ov::element::u8};

const std::vector<ov::test::TileSpecificParams> repeats = {
        // tile by single axes
        {1, 1, 1, 5},
        {1, 1, 5},
        {1, 5, 1},
        {5, 1, 1},
        {1, 8},
        {8, 1},

        // tile by multiple axes
        {1, 2, 3},
        {2, 3, 1},
        {3, 1, 2},

        // identical tile case
        {1, 1, 1},

        // input shapes with more than 4D is not supported by runtime yet
        // {1, 1, 1, 2, 1, 2}

        // looks like this values is too big. Test fails due result mismatch between CPU an NPU
        // {1, 1, 1, 128}, {1, 1, 128, 1}, {1, 128, 1, 1}, {128, 1, 1, 1},
};

const std::vector<std::vector<ov::Shape>> inputShapes = {
        {{2}},       {{2, 3}},

        {{3, 4, 2}}, {{2, 3, 4, 2}}, {{1, 1, 128, 1}},

        // input shapes with more than 4D is not supported by runtime yet
        // {{1, 4, 3, 1, 3, 1}}
};

INSTANTIATE_TEST_SUITE_P(
        smoke_Tile, TileLayerTest_NPU3700,
        ::testing::Combine(::testing::ValuesIn(repeats), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes)),
                           ::testing::Values(ov::test::utils::DEVICE_NPU)),
        TileLayerTest_NPU3700::getTestCaseName);

const auto tileParams = ::testing::Combine(
        ::testing::ValuesIn(std::vector<ov::test::TileSpecificParams>({{1, 3, 2}, {3, 2, 1, 5}, {1, 3, 2, 1}})),
        ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                std::vector<std::vector<ov::Shape>>({{{1, 4, 3, 2}}, {{4, 3, 2, 1}}, {{4, 3, 2, 5}}}))),
        ::testing::Values(ov::test::utils::DEVICE_NPU));

const auto tileParamsPrecommit = ::testing::Combine(
        ::testing::ValuesIn(std::vector<ov::test::TileSpecificParams>({{2, 3, 1}})), ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                ov::test::static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{3, 4, 2}}}))),
        ::testing::Values(ov::test::utils::DEVICE_NPU));

// NPU3720

INSTANTIATE_TEST_SUITE_P(smoke_Tile, TileLayerTest_NPU3720, tileParams, TileLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Tile, TileLayerTest_NPU3720, tileParamsPrecommit,
                         TileLayerTest_NPU3720::getTestCaseName);

// NPU4000

INSTANTIATE_TEST_SUITE_P(smoke_Tile, TileLayerTest_NPU4000, tileParams, TileLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Tile, TileLayerTest_NPU4000, tileParamsPrecommit,
                         TileLayerTest_NPU4000::getTestCaseName);

// NPU3720 - tiling

// case 1: tile on two dimensions
INSTANTIATE_TEST_SUITE_P(
        smoke_Tile_tiling_1, TileLayerTest_NPU3720_tiling,
        ::testing::Combine(::testing::ValuesIn(std::vector<ov::test::TileSpecificParams>({{1, 2, 3,
                                                                                           3}})),  // repeats_values
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                   std::vector<std::vector<ov::Shape>>({{{1, 1, 2880, 50}}}))),  // input_shape
                           ::testing::Values(ov::test::utils::DEVICE_NPU)),
        TileLayerTest_NPU3720_tiling::getTestCaseName);

// case 2: repeats values aren't 1
INSTANTIATE_TEST_SUITE_P(
        smoke_Tile_tiling_2, TileLayerTest_NPU3720_tiling,
        ::testing::Combine(::testing::ValuesIn(std::vector<ov::test::TileSpecificParams>({{2, 2, 3,
                                                                                           3}})),  // repeats_values
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                   std::vector<std::vector<ov::Shape>>({{{3, 2, 723, 25}}}))),  // input_shape
                           ::testing::Values(ov::test::utils::DEVICE_NPU)),
        TileLayerTest_NPU3720_tiling::getTestCaseName);

// case 3: repeats values may be 1
INSTANTIATE_TEST_SUITE_P(
        smoke_Tile_tiling_3, TileLayerTest_NPU3720_tiling,
        ::testing::Combine(::testing::ValuesIn(std::vector<ov::test::TileSpecificParams>({{3, 1, 3,
                                                                                           2}})),  // repeats_values
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                   std::vector<std::vector<ov::Shape>>({{{2, 3, 360, 50}}}))),  // input_shape
                           ::testing::Values(ov::test::utils::DEVICE_NPU)),
        TileLayerTest_NPU3720_tiling::getTestCaseName);

// case 4: tiling dim not divisible
INSTANTIATE_TEST_SUITE_P(
        smoke_Tile_tiling_4, TileLayerTest_NPU3720_tiling,
        ::testing::Combine(::testing::ValuesIn(std::vector<ov::test::TileSpecificParams>({{1, 1, 1,
                                                                                           14}})),  // repeats_values
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                   std::vector<std::vector<ov::Shape>>({{{1, 300, 2744, 1}}}))),  // input_shape
                           ::testing::Values(ov::test::utils::DEVICE_NPU)),
        TileLayerTest_NPU3720_tiling::getTestCaseName);

// model case: tensor<1x32x1x1xf16> -> tensor<1x32x1x65536xf16> , NHWC
INSTANTIATE_TEST_SUITE_P(
        smoke_Tile_tiling_5, TileLayerTest_NPU3720_tiling,
        ::testing::Combine(::testing::ValuesIn(std::vector<ov::test::TileSpecificParams>({{1, 1, 1,
                                                                                           65536}})),  // repeats_values
                           ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                   std::vector<std::vector<ov::Shape>>({{{1, 32, 1, 1}}}))),  // input_shape
                           ::testing::Values(ov::test::utils::DEVICE_NPU)),
        TileLayerTest_NPU3720_tiling::getTestCaseName);

// case 6: INT32 tiling on two dimensions
INSTANTIATE_TEST_SUITE_P(
        smoke_Tile_tiling_6, TileLayerTest_NPU3720_tiling,
        ::testing::Combine(::testing::ValuesIn(std::vector<ov::test::TileSpecificParams>({{1, 1, 256,
                                                                                           256}})),  // repeats_values
                           ::testing::Values(ov::element::i32),
                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                   std::vector<std::vector<ov::Shape>>({{{1, 32, 1, 1}}}))),  // input_shape
                           ::testing::Values(ov::test::utils::DEVICE_NPU)),
        TileLayerTest_NPU3720_tiling::getTestCaseName);

}  // namespace
