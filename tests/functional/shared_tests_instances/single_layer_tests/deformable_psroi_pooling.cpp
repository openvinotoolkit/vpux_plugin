// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include <common/functions.h>
#include "common_test_utils/test_constants.hpp"
#include "single_op_tests/deformable_psroi_pooling.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class DeformablePSROIPoolingLayerTestCommon : public DeformablePSROIPoolingLayerTest, virtual public VpuOv2LayerTest {};

class DeformablePSROIPoolingLayerTest_NPU3700 : public DeformablePSROIPoolingLayerTestCommon {};

TEST_P(DeformablePSROIPoolingLayerTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const auto deformablePSROIParams = ::testing::Combine(
        ::testing::Values(2),                                                                    // output_dim
        ::testing::Values(2),                                                                    // group_size
        ::testing::ValuesIn(std::vector<float>{1.0, 0.0625}),                                    // spatial scale
        ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{1, 1}, {2, 2}, {3, 3}, {4, 4}}),  // spatial_bins_x_y
        ::testing::ValuesIn(std::vector<float>{0.0, 0.01, 0.1}),                                 // trans_std
        ::testing::Values(2));

std::vector<std::vector<ov::Shape>> shapesStatic{// dataShape, roisShape, offsetsShape
                                                 {{1, 8, 16, 16}, {10, 5}, {10, 2, 2, 2}},
                                                 {{1, 8, 67, 32}, {10, 5}, {10, 2, 2, 2}}};

const auto deformablePSROICases_test_params =
        ::testing::Combine(deformablePSROIParams,
                           ::testing::ValuesIn(static_shapes_to_test_representation(shapesStatic)),  // data input shape
                           ::testing::Values(ov::element::f32),                                      // Net precision
                           ::testing::Values(DEVICE_NPU));                                           // Device name

INSTANTIATE_TEST_SUITE_P(smoke_TestsDeformablePSROIPooling, DeformablePSROIPoolingLayerTest_NPU3700,
                         deformablePSROICases_test_params, DeformablePSROIPoolingLayerTest_NPU3700::getTestCaseName);

const auto deformablePSROIParams_advanced =
        ::testing::Combine(::testing::Values(49),                                           // output_dim
                           ::testing::Values(3),                                            // group_size
                           ::testing::ValuesIn(std::vector<float>{0.0625}),                 // spatial scale
                           ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{4, 4}}),  // spatial_bins_x_y
                           ::testing::ValuesIn(std::vector<float>{0.1}),                    // trans_std
                           ::testing::Values(3));                                           // part_size

std::vector<std::vector<ov::Shape>> shapesStaticAdvanced{// dataShape, roisShape, offsetsShape
                                                         {{1, 441, 8, 8}, {30, 5}, {30, 2, 3, 3}}};
const auto deformablePSROICases_test_params_advanced = ::testing::Combine(
        deformablePSROIParams_advanced,
        ::testing::ValuesIn(static_shapes_to_test_representation(shapesStaticAdvanced)),  // data input shape
        ::testing::Values(ov::element::f32),                                              // Net precision
        ::testing::Values(DEVICE_NPU));                                                   // Device name

INSTANTIATE_TEST_SUITE_P(smoke_TestsDeformablePSROIPooling_advanced, DeformablePSROIPoolingLayerTest_NPU3700,
                         deformablePSROICases_test_params_advanced,
                         DeformablePSROIPoolingLayerTest_NPU3700::getTestCaseName);

const auto deformablePSROIParams_advanced1 =
        ::testing::Combine(::testing::Values(49),                                                   // output_dim
                           ::testing::Values(3),                                                    // group_size
                           ::testing::ValuesIn(std::vector<float>{0.0625}),                         // spatial scale
                           ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{1, 1}, {4, 4}}),  // spatial_bins_x_y
                           ::testing::ValuesIn(std::vector<float>{0.0, 0.1}),                       // trans_std
                           ::testing::Values(3));                                                   // part_size

std::vector<std::vector<ov::Shape>> shapesStaticAdvanced1{// dataShape, roisShape, offsetsShape
                                                          {{1, 441, 8, 8}, {30, 5}}};

const auto deformablePSROICases_test_params_advanced1 = ::testing::Combine(
        deformablePSROIParams_advanced1,
        ::testing::ValuesIn(static_shapes_to_test_representation(shapesStaticAdvanced1)),  // data input shape
        ::testing::Values(ov::element::f32),                                               // Net precision
        ::testing::Values(DEVICE_NPU));                                                    // Device name

INSTANTIATE_TEST_SUITE_P(smoke_TestsDeformablePSROIPooling_advanced1, DeformablePSROIPoolingLayerTest_NPU3700,
                         deformablePSROICases_test_params_advanced1,
                         DeformablePSROIPoolingLayerTest_NPU3700::getTestCaseName);

}  // namespace
