//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "single_op_tests/broadcast.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class BroadcastLayerTestCommon : public BroadcastLayerTest, virtual public VpuOv2LayerTest {};

class BroadcastLayerTest_NPU3700 : public BroadcastLayerTestCommon {};
class BroadcastLayerTest_NPU3720 : public BroadcastLayerTestCommon {};
class BroadcastLayerTest_NPU4000 : public BroadcastLayerTestCommon {};

TEST_P(BroadcastLayerTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

TEST_P(BroadcastLayerTest_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(BroadcastLayerTest_NPU4000, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

// NUMPY MODE

const std::vector<ov::element::Type> inputPrecision = {ov::element::f16, ov::element::f32};

std::vector<std::vector<ov::Shape>> inShapesNumpy = {{{3, 1}}, {{1, 4, 1}}};

std::vector<std::vector<size_t>> targetShapesNumpy = {{2, 3, 6}, {1, 4, 4}};

const auto numpyBroadcastParams1 =
        ::testing::Combine(::testing::Values(targetShapesNumpy[0]),          // target shape
                           ::testing::Values(ov::AxisSet{}),                 // not used in numpy mode
                           ::testing::Values(ov::op::BroadcastType::NUMPY),  // broadcast mode
                           ::testing::Values(static_shapes_to_test_representation(inShapesNumpy[0])),  // Input shape
                           ::testing::ValuesIn(inputPrecision),                                        // Model type
                           ::testing::Values(DEVICE_NPU));                                             // Device name

const auto numpyBroadcastParams2 =
        ::testing::Combine(::testing::Values(targetShapesNumpy[1]),                                    // target shape
                           ::testing::Values(ov::AxisSet{}),                                           // axes mapping
                           ::testing::Values(ov::op::BroadcastType::NUMPY),                            // broadcast mode
                           ::testing::Values(static_shapes_to_test_representation(inShapesNumpy[1])),  // Input shape
                           ::testing::ValuesIn(inputPrecision),                                        // Model type
                           ::testing::Values(DEVICE_NPU));                                             // Device name

// BIDIRECTIONAL MODE

std::vector<std::vector<ov::Shape>> inShapesBidi = {{{4, 1}}, {{1, 4, 1}}, {{4, 1, 1}}};

std::vector<std::vector<size_t>> targetShapesBidi = {{2, 1, 4}, {1, 4, 4}, {1, 1, 2, 2}};

const auto bidirectionalBroadcastParams1 = ::testing::Combine(
        ::testing::Values(targetShapesBidi[0]), ::testing::Values(ov::AxisSet{}),  // not used in bidirectional mode
        ::testing::Values(ov::op::BroadcastType::BIDIRECTIONAL),
        ::testing::Values(static_shapes_to_test_representation(inShapesBidi[0])), ::testing::ValuesIn(inputPrecision),
        ::testing::Values(DEVICE_NPU));

const auto bidirectionalBroadcastParams2 =
        ::testing::Combine(::testing::Values(targetShapesBidi[1]), ::testing::Values(ov::AxisSet{}),
                           ::testing::Values(ov::op::BroadcastType::BIDIRECTIONAL),
                           ::testing::Values(static_shapes_to_test_representation(inShapesBidi[1])),
                           ::testing::ValuesIn(inputPrecision), ::testing::Values(DEVICE_NPU));

const auto bidirectionalBroadcastParams3 =
        ::testing::Combine(::testing::Values(targetShapesBidi[2]), ::testing::Values(ov::AxisSet{}),
                           ::testing::Values(ov::op::BroadcastType::BIDIRECTIONAL),
                           ::testing::Values(static_shapes_to_test_representation(inShapesBidi[2])),
                           ::testing::ValuesIn(inputPrecision), ::testing::Values(DEVICE_NPU));

// EXPLICIT MODE

std::vector<std::vector<ov::Shape>> inShapesExplicit = {{{3, 1}}, {{2, 4}}};

std::vector<std::vector<size_t>> targetShapesExplicit = {{2, 3, 1}, {2, 3, 4}};

std::vector<ov::AxisSet> axes = {{1, 2}, {0, 2}};

const auto explicitBroadcastParams1 =
        ::testing::Combine(::testing::Values(targetShapesExplicit[0]), ::testing::Values(axes[0]),
                           ::testing::Values(ov::op::BroadcastType::EXPLICIT),
                           ::testing::Values(static_shapes_to_test_representation(inShapesExplicit[0])),
                           ::testing::ValuesIn(inputPrecision), ::testing::Values(DEVICE_NPU));

const auto explicitBroadcastParams2 =
        ::testing::Combine(::testing::Values(targetShapesExplicit[1]), ::testing::Values(axes[1]),
                           ::testing::Values(ov::op::BroadcastType::EXPLICIT),
                           ::testing::Values(static_shapes_to_test_representation(inShapesExplicit[1])),
                           ::testing::ValuesIn(inputPrecision), ::testing::Values(DEVICE_NPU));

// ------ NPU3700 ------

INSTANTIATE_TEST_CASE_P(smoke_NumpyBroadcastCheck1, BroadcastLayerTest_NPU3700, numpyBroadcastParams1,
                        BroadcastLayerTest_NPU3700::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_NumpyBroadcastCheck2, BroadcastLayerTest_NPU3700, numpyBroadcastParams2,
                        BroadcastLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BidirectionalBroadcastCheck1, BroadcastLayerTest_NPU3700, bidirectionalBroadcastParams1,
                        BroadcastLayerTest_NPU3700::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_BidirectionalBroadcastCheck2, BroadcastLayerTest_NPU3700, bidirectionalBroadcastParams2,
                        BroadcastLayerTest_NPU3700::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_BidirectionalBroadcastCheck3, BroadcastLayerTest_NPU3700, bidirectionalBroadcastParams3,
                        BroadcastLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ExplicitBroadcastCheck1, BroadcastLayerTest_NPU3700, explicitBroadcastParams1,
                        BroadcastLayerTest_NPU3700::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_ExplicitBroadcastCheck2, BroadcastLayerTest_NPU3700, explicitBroadcastParams2,
                        BroadcastLayerTest_NPU3700::getTestCaseName);

// ------ NPU3720 ------

INSTANTIATE_TEST_CASE_P(smoke_precommit_NumpyBroadcastCheck1, BroadcastLayerTest_NPU3720, numpyBroadcastParams1,
                        BroadcastLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_NumpyBroadcastCheck2, BroadcastLayerTest_NPU3720, numpyBroadcastParams2,
                        BroadcastLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_BidirectionalBroadcastCheck1, BroadcastLayerTest_NPU3720,
                        bidirectionalBroadcastParams1, BroadcastLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_BidirectionalBroadcastCheck2, BroadcastLayerTest_NPU3720, bidirectionalBroadcastParams2,
                        BroadcastLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_BidirectionalBroadcastCheck3, BroadcastLayerTest_NPU3720, bidirectionalBroadcastParams3,
                        BroadcastLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_ExplicitBroadcastCheck1, BroadcastLayerTest_NPU3720, explicitBroadcastParams1,
                        BroadcastLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_ExplicitBroadcastCheck2, BroadcastLayerTest_NPU3720, explicitBroadcastParams2,
                        BroadcastLayerTest_NPU3720::getTestCaseName);

// ------ NPU4000 ------

INSTANTIATE_TEST_CASE_P(smoke_precommit_NumpyBroadcastCheck1, BroadcastLayerTest_NPU4000, numpyBroadcastParams1,
                        BroadcastLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_NumpyBroadcastCheck2, BroadcastLayerTest_NPU4000, numpyBroadcastParams2,
                        BroadcastLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_BidirectionalBroadcastCheck1, BroadcastLayerTest_NPU4000,
                        bidirectionalBroadcastParams1, BroadcastLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_BidirectionalBroadcastCheck2, BroadcastLayerTest_NPU4000, bidirectionalBroadcastParams2,
                        BroadcastLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_BidirectionalBroadcastCheck3, BroadcastLayerTest_NPU4000, bidirectionalBroadcastParams3,
                        BroadcastLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_ExplicitBroadcastCheck1, BroadcastLayerTest_NPU4000, explicitBroadcastParams1,
                        BroadcastLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_ExplicitBroadcastCheck2, BroadcastLayerTest_NPU4000, explicitBroadcastParams2,
                        BroadcastLayerTest_NPU4000::getTestCaseName);

}  // namespace
