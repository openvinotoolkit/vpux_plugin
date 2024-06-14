// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <common/functions.h>
#include "single_op_tests/adaptive_pooling.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class AdaPoolLayerTestCommon : public AdaPoolLayerTest, virtual public VpuOv2LayerTest {};
class AdaPoolLayerTest_NPU3700 : public AdaPoolLayerTestCommon {};
class AdaPoolLayerTest_NPU3720 : public AdaPoolLayerTestCommon {};
class AdaPoolLayerTest_NPU4000 : public AdaPoolLayerTestCommon {};

TEST_P(AdaPoolLayerTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

TEST_P(AdaPoolLayerTest_NPU3720, SW) {
    abs_threshold = 0.05;
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(AdaPoolLayerTest_NPU4000, SW) {
    abs_threshold = 0.05;
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f16,
        ov::element::f32,

};

std::vector<std::vector<ov::Shape>> inShape3DCases = {{{2, 3, 7}}, {{1, 1, 3}}};
std::vector<std::vector<ov::Shape>> inShape4DCases = {{{1, 3, 32, 32}}, {{1, 1, 3, 2}}};
std::vector<std::vector<ov::Shape>> inShape5DCases = {{{1, 17, 4, 5, 4}}, {{1, 1, 3, 2, 3}}};

/* ============= 3D/4D/5D AdaptivePool NPU3700 ============= */
const auto AdaPool3DCases =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShape3DCases)),  // inputShape
                           ::testing::ValuesIn(std::vector<std::vector<int>>{{1}, {3}}),  // pooledSpatialShape
                           ::testing::ValuesIn(std::vector<std::string>{"max", "avg"}),   // mode
                           ::testing::ValuesIn(netPrecisions),                            // precision
                           ::testing::Values(DEVICE_NPU));                                // device

const auto AdaPool4DCases =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShape4DCases)),  // inputShape
                           ::testing::ValuesIn(std::vector<std::vector<int>>{{3, 5}, {16, 16}}),  // pooledSpatialShape
                           ::testing::ValuesIn(std::vector<std::string>{"max", "avg"}),           // mode
                           ::testing::ValuesIn(netPrecisions),                                    // precision
                           ::testing::Values(DEVICE_NPU));                                        // device

const auto AdaPool5DCases = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(inShape5DCases)),  // inputShape
        ::testing::ValuesIn(std::vector<std::vector<int>>{{1, 1, 1}, {3, 5, 3}}),   // pooledSpatialShape
        ::testing::ValuesIn(std::vector<std::string>{"max", "avg"}),                // mode
        ::testing::ValuesIn(netPrecisions),                                         // precision
        ::testing::Values(DEVICE_NPU));                                             // device

// ------ NPU3700 ------

INSTANTIATE_TEST_CASE_P(smoke_TestsAdaPool3D, AdaPoolLayerTest_NPU3700, AdaPool3DCases,
                        AdaPoolLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_TestsAdaPool4D, AdaPoolLayerTest_NPU3700, AdaPool4DCases,
                        AdaPoolLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_TestsAdaPool5D, AdaPoolLayerTest_NPU3700, AdaPool5DCases,
                        AdaPoolLayerTest_NPU3700::getTestCaseName);

/* ============= 3D/4D AdaptivePool NPU3720/NPU4000 ============= */

std::vector<std::vector<ov::Shape>> inShape3DSingleCases = {{{2, 3, 7}}};
std::vector<std::vector<ov::Shape>> inShape4DSingleCases = {{{1, 128, 32, 64}}};

const auto AdaPoolCase3D =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShape3DSingleCases)),
                           ::testing::ValuesIn(std::vector<std::vector<int>>{{1}, {3}}),
                           ::testing::ValuesIn(std::vector<std::string>{"avg", "max"}),
                           ::testing::ValuesIn(netPrecisions), ::testing::Values(DEVICE_NPU));

const auto AdaPoolCase4D =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShape4DSingleCases)),
                           ::testing::ValuesIn(std::vector<std::vector<int>>{{2, 2}, {3, 3}, {6, 6}}),
                           ::testing::ValuesIn(std::vector<std::string>{"avg", "max"}),
                           ::testing::ValuesIn(netPrecisions), ::testing::Values(DEVICE_NPU));

// ------ NPU3720 ------

INSTANTIATE_TEST_CASE_P(smoke_precommit_TestsAdaPool3D, AdaPoolLayerTest_NPU3720, AdaPoolCase3D,
                        AdaPoolLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_TestsAdaPool4D, AdaPoolLayerTest_NPU3720, AdaPoolCase4D,
                        AdaPoolLayerTest_NPU3720::getTestCaseName);

// ------ NPU4000 ------

INSTANTIATE_TEST_CASE_P(smoke_precommit_TestsAdaPool3D, AdaPoolLayerTest_NPU4000, AdaPoolCase3D,
                        AdaPoolLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_TestsAdaPool4D, AdaPoolLayerTest_NPU4000, AdaPoolCase4D,
                        AdaPoolLayerTest_NPU4000::getTestCaseName);

}  // namespace
