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

class ConversionLayerTest_NPU3700 : public ConversionLayerTestCommon {};
class ConversionLayerTest_NPU3720 : public ConversionLayerTestCommon {};
using ConversionLayerTest_NPU3720_ELF = ConversionLayerTest_NPU3720;
class ConversionLayerTest_NPU4000 : public ConversionLayerTestCommon {};

TEST_P(ConversionLayerTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

TEST_P(ConversionLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ConversionLayerTest_NPU3720_ELF, HW) {
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

const std::vector<ov::element::Type> netPrecisions_NPU3700 = {ov::element::f32, ov::element::f16, ov::element::u8};

const std::vector<ov::element::Type> netPrecisions = {ov::element::f32, ov::element::f16, ov::element::u8,
                                                      ov::element::i8, ov::element::i32};

const auto configParams =
        ::testing::Combine(::testing::ValuesIn(conversionOpTypes),                              // Conversion type
                           ::testing::ValuesIn(static_shapes_to_test_representation(inShape)),  // Input shapes
                           ::testing::ValuesIn(netPrecisions),                                  // Input type
                           ::testing::ValuesIn(netPrecisions),                                  // Convert type
                           ::testing::Values(DEVICE_NPU));                                      // Device name

INSTANTIATE_TEST_SUITE_P(smoke_NoReshape, ConversionLayerTest_NPU3700,
                         ::testing::Combine(::testing::ValuesIn(conversionOpTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inShape)),
                                            ::testing::ValuesIn(netPrecisions_NPU3700),
                                            ::testing::ValuesIn(netPrecisions_NPU3700), ::testing::Values(DEVICE_NPU)),
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Conversion_NoReshape, ConversionLayerTest_NPU3720,
                         ::testing::Combine(::testing::ValuesIn(conversionOpTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inShape)),
                                            ::testing::ValuesIn(netPrecisions), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(DEVICE_NPU)),
                         ConversionLayerTest::getTestCaseName);

// ------ ELF ------

INSTANTIATE_TEST_SUITE_P(NoReshape, ConversionLayerTest_NPU3720_ELF,
                         ::testing::Combine(::testing::Values(ConversionTypes::CONVERT),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inShape)),
                                            ::testing::ValuesIn(netPrecisions), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(DEVICE_NPU)),
                         ConversionLayerTest::getTestCaseName);

// ------ NPU4000 ------

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Conversion, ConversionLayerTest_NPU4000, configParams,
                         ConversionLayerTest::getTestCaseName);

}  // namespace
