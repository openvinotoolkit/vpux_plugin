//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <common/functions.h>
#include "single_op_tests/roi_align.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {

namespace test {

class ROIAlignLayerTest_NPU3700 : public ROIAlignLayerTest, virtual public VpuOv2LayerTest {};

class ROIAlignV9LayerTest_NPU3720 : public ROIAlignV9LayerTest, virtual public VpuOv2LayerTest {};

class ROIAlignV9LayerTest_NPU4000 : public ROIAlignV9LayerTest, virtual public VpuOv2LayerTest {};

TEST_P(ROIAlignLayerTest_NPU3700, HW) {
    setSkipInferenceCallback([this](std::stringstream& skip) {
        if (getBackendName(*ov::test::utils::PluginCache::get().core()) == "LEVEL0") {
            skip << "Bad results on Level0";
        }
    });
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

TEST_P(ROIAlignV9LayerTest_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ROIAlignV9LayerTest_NPU4000, SW) {
    abs_threshold = 0.03;
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test

}  // namespace ov

using namespace ov::test;

namespace {
const std::vector<ov::element::Type> modelTypes = {
        ov::element::f16,
};

const std::vector<std::string> poolingMode = {"avg", "max"};

//
// NPU3700
//
std::vector<std::vector<ov::Shape>> inputShape_3700 = {
        {{2, 18, 20, 20}}, {{2, 4, 20, 20}}, {{2, 4, 20, 40}}, {{10, 1, 20, 20}}};

const std::vector<ov::Shape> coordsShape_3700 = {{2, 4}};

const std::vector<int> pooledH_3700 = {2};

const std::vector<int> pooledW_3700 = {2};

const std::vector<float> spatialScale_3700 = {0.625f, 1.0f};

const std::vector<int> poolingRatio_3700 = {2};

const auto testROIAlignParams = testing::Combine(
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShape_3700)),
        testing::ValuesIn(coordsShape_3700), testing::ValuesIn(pooledH_3700), testing::ValuesIn(pooledW_3700),
        testing::ValuesIn(spatialScale_3700), testing::ValuesIn(poolingRatio_3700), testing::ValuesIn(poolingMode),
        testing::ValuesIn(modelTypes), testing::Values(DEVICE_NPU));

//
// NPU3720/4000
//
std::vector<std::vector<ov::Shape>> inputShape = {
        {{2, 22, 20, 20}}, {{2, 18, 20, 20}}, {{2, 4, 20, 20}}, {{2, 4, 20, 40}}};

const std::vector<ov::Shape> coordsShape = {{2, 4}};

const std::vector<int> pooledH = {8};

const std::vector<int> pooledW = {8};

const std::vector<float> spatialScale = {0.03125f, 1.f};

const std::vector<int> poolingRatio = {2};

const std::vector<std::string> alignedMode = {"asymmetric", "half_pixel", "half_pixel_for_nn"};

const auto testROIAlignV9Params = testing::Combine(
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShape)), testing::ValuesIn(coordsShape),
        testing::ValuesIn(pooledH), testing::ValuesIn(pooledW), testing::ValuesIn(spatialScale),
        testing::ValuesIn(poolingRatio), testing::ValuesIn(poolingMode), testing::ValuesIn(alignedMode),
        testing::ValuesIn(modelTypes), testing::Values(DEVICE_NPU));

// ------ NPU3700 ------

INSTANTIATE_TEST_SUITE_P(smoke_ROIAlign, ROIAlignLayerTest_NPU3700, testROIAlignParams,
                         ROIAlignLayerTest_NPU3700::getTestCaseName);

// ------ NPU3720 ------

INSTANTIATE_TEST_SUITE_P(precommit_ROIAlign, ROIAlignV9LayerTest_NPU3720, testROIAlignV9Params,
                         ROIAlignV9LayerTest_NPU3720::getTestCaseName);

// ------ NPU4000 ------

INSTANTIATE_TEST_SUITE_P(precommit_ROIAlign, ROIAlignV9LayerTest_NPU4000, testROIAlignV9Params,
                         ROIAlignV9LayerTest_NPU4000::getTestCaseName);

}  // namespace
