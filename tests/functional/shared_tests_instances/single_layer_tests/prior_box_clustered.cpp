//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/prior_box_clustered.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class PriorBoxClusteredLayerTestCommon : public PriorBoxClusteredLayerTest, virtual public VpuOv2LayerTest {};

class PriorBoxClusteredLayerTest_NPU3720 : public PriorBoxClusteredLayerTestCommon {};
class PriorBoxClusteredLayerTest_NPU4000 : public PriorBoxClusteredLayerTestCommon {};

TEST_P(PriorBoxClusteredLayerTest_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(PriorBoxClusteredLayerTest_NPU4000, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<std::vector<float>> widths = {{5.12f, 14.6f, 13.5f}, {7.0f, 8.2f, 33.39f}};

const std::vector<std::vector<float>> heights = {{15.12f, 15.6f, 23.5f}, {10.0f, 16.2f, 36.2f}};

const std::vector<float> step_widths = {2.0f};

const std::vector<float> step_heights = {1.5f};

const std::vector<float> step = {1.5f};

const std::vector<float> offsets = {0.5f};

const std::vector<std::vector<float>> variances = {
        {0.1f, 0.1f, 0.2f, 0.2f},
};

const std::vector<bool> clips = {true, false};

const auto layerSpeficParams =
        testing::Combine(testing::ValuesIn(widths), testing::ValuesIn(heights), testing::ValuesIn(clips),
                         testing::ValuesIn(step_widths), testing::ValuesIn(step_heights), testing::ValuesIn(step),
                         testing::ValuesIn(offsets), testing::ValuesIn(variances));

const auto params = testing::Combine(
        layerSpeficParams, testing::Values(ov::element::f16),
        testing::Values(static_shapes_to_test_representation(std::vector<ov::Shape>{{4, 4}, {50, 50}})),
        testing::Values(DEVICE_NPU));

const auto precommit_layerSpeficParams =
        testing::Combine(testing::ValuesIn(std::vector<std::vector<float>>{{2.56f, 7.3f, 6.75f}}),
                         testing::ValuesIn(std::vector<std::vector<float>>{{7.56f, 7.8f, 16.75f}}),
                         testing::ValuesIn(clips), testing::ValuesIn(step_widths), testing::ValuesIn(step_heights),
                         testing::ValuesIn(step), testing::ValuesIn(offsets), testing::ValuesIn(variances));

const auto paramsPrecommit = testing::Combine(
        precommit_layerSpeficParams, testing::Values(ov::element::f16),
        testing::Values(static_shapes_to_test_representation(std::vector<ov::Shape>{{4, 4}, {13, 13}})),
        testing::Values(DEVICE_NPU));

// ------ NPU3720 ------

INSTANTIATE_TEST_CASE_P(smoke_PriorBoxClustered, PriorBoxClusteredLayerTest_NPU3720, params,
                        PriorBoxClusteredLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_PriorBoxClustered, PriorBoxClusteredLayerTest_NPU3720, paramsPrecommit,
                        PriorBoxClusteredLayerTest_NPU3720::getTestCaseName);

// ------ NPU4000 ------

INSTANTIATE_TEST_CASE_P(smoke_PriorBoxClustered, PriorBoxClusteredLayerTest_NPU4000, params,
                        PriorBoxClusteredLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_PriorBoxClustered, PriorBoxClusteredLayerTest_NPU4000, paramsPrecommit,
                        PriorBoxClusteredLayerTest_NPU4000::getTestCaseName);

}  // namespace
