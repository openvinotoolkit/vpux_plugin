// Copyright (C) 2019-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/conversion.hpp"
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class ConversionLayerTestCommon : public ConversionLayerTest, virtual public VpuOv2LayerTest {};

class ConversionLayerTest_NPU3720 : public ConversionLayerTestCommon {};
class ConversionLayerTest_NPU4000 : public ConversionLayerTestCommon {};

TEST_P(ConversionLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ConversionLayerTest_NPU4000, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {
const std::vector<ConversionTypes> conversionOpTypes = {
        ConversionTypes::CONVERT,
        ConversionTypes::CONVERT_LIKE,
};

const std::vector<std::vector<ov::Shape>> inShape = {{{1, 2, 3, 4}}};

const std::vector<std::vector<ov::Shape>> inShapeTiling = {{{2000, 2000}}};

const std::vector<std::vector<ov::Shape>> inShapeOdd = {{{1, 1, 1, 111}}};

const std::vector<ov::element::Type> netPrecisions = {ov::element::f32, ov::element::f16, ov::element::u8,
                                                      ov::element::i8, ov::element::i32};

const auto configParams =
        ::testing::Combine(::testing::ValuesIn(conversionOpTypes),                              // Conversion type
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShape)),  // Input shapes
                           ::testing::ValuesIn(netPrecisions),                                  // Input type
                           ::testing::ValuesIn(netPrecisions),                                  // Convert type
                           ::testing::Values(DEVICE_NPU));

const auto configParamsU4Tiling =
        ::testing::Combine(::testing::ValuesIn(conversionOpTypes),                                    // Conversion type
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapeTiling)),  // Input shapes
                           ::testing::Values(ov::element::u4),                                        // Input type
                           ::testing::ValuesIn({ov::element::f16, ov::element::u8, ov::element::i8}),  // Convert type
                           ::testing::Values(DEVICE_NPU));                                             // Device name

const auto configParamsI4Tiling =
        ::testing::Combine(::testing::ValuesIn(conversionOpTypes),                                    // Conversion type
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapeTiling)),  // Input shapes
                           ::testing::Values(ov::element::i4),                                        // Input type
                           ::testing::ValuesIn({ov::element::i8, ov::element::f16}),                  // Convert type
                           ::testing::Values(DEVICE_NPU));

const auto configParamsU4OddShape =
        ::testing::Combine(::testing::ValuesIn(conversionOpTypes),                                 // Conversion type
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShapeOdd)),  // Input shapes
                           ::testing::Values(ov::element::u4),                                     // Input type
                           ::testing::ValuesIn({ov::element::f16, ov::element::u8, ov::element::i8}),  // Convert type
                           ::testing::Values(DEVICE_NPU));

// ------ NPU3720 ------

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Conversion, ConversionLayerTest_NPU3720, configParams,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_i4_Conversion, ConversionLayerTest_NPU3720, configParamsI4Tiling,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_u4_Conversion, ConversionLayerTest_NPU3720, configParamsU4Tiling,
                         ConversionLayerTest::getTestCaseName);

// Tracking number [E#128077]
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_precommit_u4_odd_Conversion, ConversionLayerTest_NPU3720,
                         configParamsU4OddShape, ConversionLayerTest::getTestCaseName);

// ------ NPU4000 ------

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Conversion, ConversionLayerTest_NPU4000, configParams,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_i4_Conversion, ConversionLayerTest_NPU4000, configParamsI4Tiling,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_u4_Conversion, ConversionLayerTest_NPU4000, configParamsU4Tiling,
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_u4_odd_Conversion, ConversionLayerTest_NPU4000, configParamsU4OddShape,
                         ConversionLayerTest::getTestCaseName);

}  // namespace
