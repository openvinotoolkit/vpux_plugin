// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <common/functions.h>
#include "single_op_tests/gather_elements.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {
class GatherElementsLayerTestCommon : public GatherElementsLayerTest, virtual public VpuOv2LayerTest {};

class GatherElementsLayerTest_NPU3700 : public GatherElementsLayerTestCommon {};

class GatherElementsLayerTest_NPU3720 : public GatherElementsLayerTestCommon {};
class GatherElementsLayerTest_NPU4000 : public GatherElementsLayerTestCommon {};

TEST_P(GatherElementsLayerTest_NPU3700, HW) {
    setSkipInferenceCallback([this](std::stringstream& skip) {
        if (getBackendName(*ov::test::utils::PluginCache::get().core()) == "LEVEL0") {
            skip << "Bad results on Level0";
        }
    });
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

TEST_P(GatherElementsLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(GatherElementsLayerTest_NPU4000, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> dPrecisions = {ov::element::f32};

const std::vector<ov::element::Type> iPrecisions = {ov::element::i32};

const std::vector<int> axes_set1 = {-1, 0, 1};
const std::vector<int> axes_set2 = {-2, 1};
const std::vector<int> axes_set3 = {0};

const std::vector<std::vector<ov::Shape>> iShapes = {{{2, 2}}, {{5, 7, 9, 1}}, {{2, 2, 1}}};

// ------ NPU3700 ------
INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_set1, GatherElementsLayerTest_NPU3700,
                         testing::Combine(testing::ValuesIn({static_shapes_to_test_representation(iShapes[0])}),
                                          testing::Values(ov::Shape{2, 2}), testing::ValuesIn(axes_set1),
                                          testing::ValuesIn(dPrecisions), testing::ValuesIn(iPrecisions),
                                          testing::Values(DEVICE_NPU)),
                         GatherElementsLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_set2, GatherElementsLayerTest_NPU3700,
                         testing::Combine(testing::ValuesIn({static_shapes_to_test_representation(iShapes[1])}),
                                          testing::Values(ov::Shape{5, 7, 9, 1}), testing::ValuesIn(axes_set2),
                                          testing::ValuesIn(dPrecisions), testing::ValuesIn(iPrecisions),
                                          testing::Values(DEVICE_NPU)),
                         GatherElementsLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_set3, GatherElementsLayerTest_NPU3700,
                         testing::Combine(testing::ValuesIn({static_shapes_to_test_representation(iShapes[2])}),
                                          testing::Values(ov::Shape{4, 2, 1}), testing::ValuesIn(axes_set3),
                                          testing::ValuesIn(dPrecisions), testing::ValuesIn(iPrecisions),
                                          testing::Values(DEVICE_NPU)),
                         GatherElementsLayerTest_NPU3700::getTestCaseName);

// ------ NPU3720/4000 ------
const auto GatherElements_PRECOMMIT_set1 =
        ::testing::Combine(testing::ValuesIn({static_shapes_to_test_representation(iShapes[0])}),
                           testing::Values(ov::Shape{2, 2}), testing::ValuesIn(axes_set1),
                           testing::ValuesIn(dPrecisions), testing::ValuesIn(iPrecisions), testing::Values(DEVICE_NPU));

const auto GatherElements_PRECOMMIT_set2 =
        ::testing::Combine(testing::ValuesIn({static_shapes_to_test_representation(iShapes[1])}),
                           testing::Values(ov::Shape{5, 7, 9, 1}), testing::ValuesIn(axes_set2),
                           testing::ValuesIn(dPrecisions), testing::ValuesIn(iPrecisions), testing::Values(DEVICE_NPU));

const auto GatherElements_PRECOMMIT_set3 = ::testing::Combine(
        ::testing::ValuesIn({static_shapes_to_test_representation(iShapes[2])}), ::testing::Values(ov::Shape{4, 2, 1}),
        ::testing::ValuesIn(axes_set3), ::testing::ValuesIn(dPrecisions), ::testing::ValuesIn(iPrecisions),
        ::testing::Values(DEVICE_NPU));

// ------ NPU3720 ------
INSTANTIATE_TEST_SUITE_P(smoke_precommit_GatherElements_set1, GatherElementsLayerTest_NPU3720,
                         GatherElements_PRECOMMIT_set1, GatherElementsLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GatherElements_set2, GatherElementsLayerTest_NPU3720,
                         GatherElements_PRECOMMIT_set2, GatherElementsLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GatherElements_set3, GatherElementsLayerTest_NPU3720,
                         GatherElements_PRECOMMIT_set3, GatherElementsLayerTest_NPU3720::getTestCaseName);

// ------ NPU4000 ------
INSTANTIATE_TEST_SUITE_P(smoke_precommit_GatherElements_set1, GatherElementsLayerTest_NPU4000,
                         GatherElements_PRECOMMIT_set1, GatherElementsLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GatherElements_set2, GatherElementsLayerTest_NPU4000,
                         GatherElements_PRECOMMIT_set2, GatherElementsLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GatherElements_set3, GatherElementsLayerTest_NPU4000,
                         GatherElements_PRECOMMIT_set3, GatherElementsLayerTest_NPU4000::getTestCaseName);

}  // namespace
