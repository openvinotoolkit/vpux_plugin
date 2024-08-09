//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <common/functions.h>
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "single_op_tests/roi_pooling.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {

namespace test {

class ROIPoolingLayerTestCommon : public ROIPoolingLayerTest, virtual public VpuOv2LayerTest {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        ROIPoolingTypes poolMethod = std::get<3>(GetParam());
        float spatialScale = std::get<2>(GetParam());

        inputs.clear();

        const auto is_roi_max_mode = (poolMethod == ROIPoolingTypes::ROI_MAX);

        const int height = is_roi_max_mode ? targetInputStaticShapes.front()[2] / spatialScale : 1;
        const int width = is_roi_max_mode ? targetInputStaticShapes.front()[3] / spatialScale : 1;

        VpuOv2LayerTest::generate_inputs(targetInputStaticShapes);

        const auto& funcInput = function->input(1);
        ov::Tensor tensor{funcInput.get_element_type(), funcInput.get_shape()};
        fill_data_roi(tensor, targetInputStaticShapes.front()[0] - 1, height, width, 1.0f, is_roi_max_mode);
        if (VpuOv2LayerTest::inputs.find(funcInput.get_node()->shared_from_this()) != VpuOv2LayerTest::inputs.end()) {
            VpuOv2LayerTest::inputs[funcInput.get_node()->shared_from_this()] = tensor;
        }
    }
};

class ROIPoolingLayerTest_NPU3720 : public ROIPoolingLayerTestCommon {};
class ROIPoolingLayerTest_NPU4000 : public ROIPoolingLayerTestCommon {};

TEST_P(ROIPoolingLayerTest_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ROIPoolingLayerTest_NPU4000, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test

}  // namespace ov

using namespace ov::test;

const std::vector<ov::Shape> paramShapes = {{{1, 3, 8, 8}}, {{3, 4, 50, 50}}};

const std::vector<ov::Shape> pooledShapes_max = {{{1, 1}}, {{2, 2}}, {{3, 3}}, {{6, 6}}};

const std::vector<ov::Shape> pooledShapes_bilinear = {/*{{1, 1}},*/ {{2, 2}}, {{3, 3}}, {{6, 6}}};

const std::vector<ov::Shape> coordShapes = {{{1, 5}}, /*{{3, 5}}, {{5, 5}}*/};

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const std::vector<float> spatial_scales = {0.625f, 1.f};

auto inputShapes = [](const std::vector<ov::Shape>& in1, const std::vector<ov::Shape>& in2) {
    std::vector<std::vector<ov::test::InputShape>> res;
    for (const auto& sh1 : in1)
        for (const auto& sh2 : in2)
            res.push_back(ov::test::static_shapes_to_test_representation({sh1, sh2}));
    return res;
}(paramShapes, coordShapes);

const auto test_ROIPooling_max = ::testing::Combine(
        ::testing::ValuesIn(inputShapes), ::testing::ValuesIn(pooledShapes_max), ::testing::ValuesIn(spatial_scales),
        ::testing::Values(ROIPoolingTypes::ROI_MAX), ::testing::ValuesIn(modelTypes), ::testing::Values(DEVICE_NPU));

const auto test_ROIPooling_bilinear =
        ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(pooledShapes_bilinear),
                           ::testing::Values(spatial_scales[1]), ::testing::Values(ROIPoolingTypes::ROI_BILINEAR),
                           ::testing::ValuesIn(modelTypes), ::testing::Values(DEVICE_NPU));

// --------- NPU3720 ---------
INSTANTIATE_TEST_SUITE_P(smoke_TestsROIPooling_max, ROIPoolingLayerTest_NPU3720, test_ROIPooling_max,
                         ROIPoolingLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIPooling_bilinear, ROIPoolingLayerTest_NPU3720, test_ROIPooling_bilinear,
                         ROIPoolingLayerTest_NPU3720::getTestCaseName);

// --------- NPU4000 ---------
// [Tracking number: E#93410]
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_precommit_TestsROIPooling_max, ROIPoolingLayerTest_NPU4000,
                         test_ROIPooling_max, ROIPoolingLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(TestsROIPooling_bilinear, ROIPoolingLayerTest_NPU4000, test_ROIPooling_bilinear,
                         ROIPoolingLayerTest_NPU4000::getTestCaseName);
