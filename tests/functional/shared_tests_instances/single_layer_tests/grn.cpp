//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/grn.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class GRNLayerTestCommon : public GrnLayerTest, virtual public VpuOv2LayerTest {};
class GRNLayerTest_NPU3700 : public GRNLayerTestCommon {};
class GRNLayerTest_NPU3720 : public GRNLayerTestCommon {};
class GRNLayerTest_NPU4000 : public GRNLayerTestCommon {};

TEST_P(GRNLayerTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

TEST_P(GRNLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(GRNLayerTest_NPU4000, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> modelTypes = {
        ov::element::f32,  // Testing FP32/FP16 netPrecision functionality only for small scope of
        ov::element::f16   // tests: GRNLayerTest, SplitLayerTest, CTCGreedyDecoderLayerTest
};

const std::vector<std::vector<ov::Shape>> inShapesNPU3700 = {
        {{{1, 3, 30, 30}}},
        {{{1, 24, 128, 224}}},
};

const std::vector<float> biases = {
        0.33f,
        1.1f,
};

const auto params = testing::Combine(testing::ValuesIn(modelTypes),
                                     testing::ValuesIn(static_shapes_to_test_representation(inShapesNPU3700)),
                                     testing::ValuesIn(biases), testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_GRN_test, GRNLayerTest_NPU3700, params, GrnLayerTest::getTestCaseName);

/* ============= NPU 3720 ============= */

// OV cases
const std::vector<std::vector<ov::Shape>> inShapes = {{{1, 8, 24, 64}}, {{3, 16, 1, 24}}, {{2, 16, 15, 20}}};

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GRN, GRNLayerTest_NPU3720,
                         testing::Combine(testing::ValuesIn(modelTypes),
                                          testing::ValuesIn(static_shapes_to_test_representation(inShapes)),
                                          testing::ValuesIn(biases), testing::Values(DEVICE_NPU)),
                         GRNLayerTest_NPU3720::getTestCaseName);

/* ============= NPU 4000 ============= */

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GRN, GRNLayerTest_NPU4000,
                         testing::Combine(testing::ValuesIn(modelTypes),
                                          testing::ValuesIn(static_shapes_to_test_representation(inShapes)),
                                          testing::ValuesIn(biases), testing::Values(DEVICE_NPU)),
                         GRNLayerTest_NPU4000::getTestCaseName);

}  // namespace
