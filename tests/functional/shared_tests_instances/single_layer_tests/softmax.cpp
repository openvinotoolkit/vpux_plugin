//
// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/softmax.hpp"
#include <algorithm>
#include <vector>
#include "vpu_ov2_layer_test.hpp"

namespace ov {

namespace test {

class SoftMaxLayerTestCommon : public subgraph::SoftMaxLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(SoftMaxLayerTestCommon, NPU3720_SW) {
    abs_threshold = 0.01;
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(SoftMaxLayerTestCommon, NPU3720_HW) {
    abs_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(SoftMaxLayerTestCommon, NPU4000_SW) {
    abs_threshold = 1e-3;
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(SoftMaxLayerTestCommon, NPU4000_HW) {
    abs_threshold = 1e-3;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace test

}  // namespace ov

using ov::test::SoftMaxLayerTestCommon;

namespace {

const std::vector<ov::test::ElementType> modelTypes = {
        ov::element::f16,
};

const std::vector<ov::test::ElementType> inputTypes = {
        ov::element::f16,
};

const std::vector<ov::test::ElementType> outputTypes = {
        ov::element::f16,
};

//
// Input 2D
//

const std::vector<ov::Shape> inShapes2D = {
        {1, 100}, {100, 1}, {10, 10}, {32, 76}, {72, 2},
};

const std::vector<size_t> axis2D = {0, 1};

const auto params2D = testing::Combine(
        testing::ValuesIn(modelTypes), testing::ValuesIn(inputTypes), testing::ValuesIn(outputTypes),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes2D)), testing::ValuesIn(axis2D),
        testing::Values(ov::test::utils::DEVICE_NPU), testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax2D, SoftMaxLayerTestCommon, params2D, SoftMaxLayerTestCommon::getTestCaseName);

//
// Input 3D
//

const std::vector<ov::Shape> inShapes3D = {{1, 4300, 2}, {8, 182, 182}};

const std::vector<size_t> axis3D = {2};

const auto params3D = testing::Combine(
        testing::ValuesIn(modelTypes), testing::ValuesIn(inputTypes), testing::ValuesIn(outputTypes),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes3D)), testing::ValuesIn(axis3D),
        testing::Values(ov::test::utils::DEVICE_NPU), testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax3D, SoftMaxLayerTestCommon, params3D, SoftMaxLayerTestCommon::getTestCaseName);

//
// Input 4D
//

const std::vector<ov::Shape> inShapes4D = {{1, 2, 108, 60}, {1, 12, 2, 148}, {1, 4, 1, 1}, {1, 100, 1, 1},
                                           {300, 21, 1, 1}, {1, 2, 48, 2},   {1, 3, 83, 4}};

const std::vector<size_t> axis4D = {0, 1, 2, 3};

const auto params4D = testing::Combine(
        testing::ValuesIn(modelTypes), testing::ValuesIn(inputTypes), testing::ValuesIn(outputTypes),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes4D)), testing::ValuesIn(axis4D),
        testing::Values(ov::test::utils::DEVICE_NPU), testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_SoftMax4D, SoftMaxLayerTestCommon, params4D, SoftMaxLayerTestCommon::getTestCaseName);

const auto precommit_params4D = testing::Combine(
        testing::ValuesIn(modelTypes), testing::ValuesIn(inputTypes), testing::ValuesIn(outputTypes),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation({{1, 2, 72, 10}})), testing::ValuesIn(axis4D),
        testing::Values(ov::test::utils::DEVICE_NPU), testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_SoftMax4D, SoftMaxLayerTestCommon, precommit_params4D,
                         SoftMaxLayerTestCommon::getTestCaseName);

//
// Test tiling functionality
//

const std::vector<ov::Shape> inShapes = {{1, 20, 64, 512}};
const std::vector<size_t> axis = {1};

const auto paramsTilingCases = testing::Combine(
        testing::ValuesIn(modelTypes), testing::ValuesIn(inputTypes), testing::ValuesIn(outputTypes),
        testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)), testing::ValuesIn(axis),
        testing::Values(ov::test::utils::DEVICE_NPU), testing::Values(ov::test::Config{}));

INSTANTIATE_TEST_SUITE_P(smoke_TilingSoftMax, SoftMaxLayerTestCommon, paramsTilingCases,
                         SoftMaxLayerTestCommon::getTestCaseName);

}  // namespace
