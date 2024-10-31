// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "single_op_tests/gather_nd.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
namespace ov {
namespace test {

class GatherNDLayerTestCommon : public GatherND8LayerTest, virtual public VpuOv2LayerTest {};

TEST_P(GatherNDLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(GatherNDLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> dPrecisions = {
        ov::element::i32,
};

const std::vector<ov::element::Type> iPrecisions = {
        ov::element::i32,
};

std::vector<std::vector<ov::Shape>> iShapeSubset1 = {{{2, 2}}, {{2, 3, 4}}};
const auto gatherNDArgsSubset1 =
        testing::Combine(testing::ValuesIn(static_shapes_to_test_representation(iShapeSubset1)),  // Data shape
                         testing::ValuesIn(std::vector<ov::Shape>({{2, 1}, {2, 1, 1}})),          // Indices shape
                         testing::ValuesIn(std::vector<int>({0, 1})),                             // Batch dims
                         testing::ValuesIn(dPrecisions),                                          // Model type
                         testing::ValuesIn(iPrecisions),                                          // Indices type
                         testing::Values(DEVICE_NPU));                                            // Device name

std::vector<std::vector<ov::Shape>> iShapeSubset2 = {{{2, 3, 4, 3, 17}}};
const auto gatherNDArgsSubset2 = testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(iShapeSubset2)),
        ::testing::ValuesIn(std::vector<ov::Shape>({{2, 3, 2, 3}})), ::testing::ValuesIn(std::vector<int>({0, 1, 2})),
        testing::ValuesIn(dPrecisions), testing::ValuesIn(iPrecisions), testing::Values(DEVICE_NPU));

std::vector<std::vector<ov::Shape>> iShapeSubsetPrecommit = {{{5, 7, 3}}};
const auto gatherNDArgsSubsetPrecommit = testing::Combine(
        testing::ValuesIn(static_shapes_to_test_representation(iShapeSubsetPrecommit)),
        testing::ValuesIn(std::vector<ov::Shape>({{5, 1}})), testing::ValuesIn(std::vector<int>({1})),
        testing::Values(ov::element::i32), testing::Values(ov::element::i32), testing::Values(DEVICE_NPU));

std::vector<std::vector<ov::Shape>> iShapeSubsetTiling = {{{2, 5, 128, 512}}};
const auto gatherNDArgsSubsetTiling = testing::Combine(
        testing::ValuesIn(static_shapes_to_test_representation(iShapeSubsetTiling)),
        testing::ValuesIn(std::vector<ov::Shape>({{2, 1, 100, 2}})), testing::ValuesIn(std::vector<int>({1})),
        testing::Values(ov::element::i32), testing::Values(ov::element::i32), testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_GatherND, GatherNDLayerTestCommon, gatherNDArgsSubset1,
                         GatherND8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GatherND, GatherNDLayerTestCommon, gatherNDArgsSubsetPrecommit,
                         GatherND8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_tiling_GatherND, GatherNDLayerTestCommon, gatherNDArgsSubsetTiling,
                         GatherND8LayerTest::getTestCaseName);

}  // namespace
