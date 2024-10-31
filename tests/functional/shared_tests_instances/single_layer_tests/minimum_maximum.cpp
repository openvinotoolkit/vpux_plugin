//
// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/minimum_maximum.hpp"
#include <common/functions.h>
#include <vector>
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class MaxMinLayerTestCommon : public MaxMinLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(MaxMinLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(MaxMinLayerTestCommon, NPU4000_SW) {
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

const std::vector<MinMaxOpType> opType = {
        MinMaxOpType::MINIMUM,
        MinMaxOpType::MAXIMUM,
};

const std::vector<InputLayerType> inputType = {InputLayerType::CONSTANT};

const std::vector<std::vector<ov::Shape>> inShapes3D = {{{1, 2, 4}, {1}}};
const std::vector<std::vector<ov::Shape>> inShapes4D = {{{1, 64, 32, 32}, {1, 64, 32, 32}}, {{1, 1, 1, 3}, {1}}};
const std::vector<std::vector<ov::Shape>> inShapesGeneric = {{{1, 1, 16, 32}, {1, 1, 16, 32}}, {{32}, {1}}};

const auto params0 = testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShapes4D)),
                                      ::testing::ValuesIn(opType), ::testing::ValuesIn(modelTypes),
                                      ::testing::ValuesIn(inputType), ::testing::Values(DEVICE_NPU));

const auto params1 = testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShapes3D)),
                                      ::testing::ValuesIn(opType), ::testing::ValuesIn(modelTypes),
                                      ::testing::ValuesIn(inputType), ::testing::Values(DEVICE_NPU));

const auto params2 = testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShapesGeneric)),
                                      ::testing::ValuesIn(opType), ::testing::ValuesIn(modelTypes),
                                      ::testing::ValuesIn(inputType), ::testing::Values(DEVICE_NPU));

const auto params3 = testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(
                                              std::vector<std::vector<ov::Shape>>({{{1, 1, 1, 3}, {1}}}))),
                                      ::testing::ValuesIn(opType), ::testing::ValuesIn(modelTypes),
                                      ::testing::ValuesIn(inputType), ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_Min_Max_test0, MaxMinLayerTestCommon, params0, MaxMinLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Min_Max_test1, MaxMinLayerTestCommon, params1, MaxMinLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Min_Max_test2, MaxMinLayerTestCommon, params2, MaxMinLayerTestCommon::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Min_Max_test3, MaxMinLayerTestCommon, params3, MaxMinLayerTestCommon::getTestCaseName);

}  // namespace
