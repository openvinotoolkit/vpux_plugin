// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/log_softmax.hpp"
#include <vector>
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class LogSoftmaxLayerTestCommon : public LogSoftmaxLayerTest, virtual public VpuOv2LayerTest {};
class LogSoftmaxLayerTest_NPU3720 : public LogSoftmaxLayerTestCommon {};
class LogSoftmaxLayerTest_NPU4000 : public LogSoftmaxLayerTestCommon {};

TEST_P(LogSoftmaxLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(LogSoftmaxLayerTest_NPU4000, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> modelType = {ov::element::f16};

std::vector<int64_t> axis2D = {0, 1};

/* ============= LogSoftmax NPU3720/NPU4000 ============= */

std::vector<std::vector<ov::Shape>> inShapes2D = {{{12, 5}},    {{1, 40}},   {{1, 66}},   {{1, 72}},   {{5, 120}},
                                                  {{5, 59}},    {{64, 29}},  {{1, 2312}}, {{1, 4192}}, {{1, 4335}},
                                                  {{10, 6495}}, {{1200, 5}}, {{2708, 7}}};

std::vector<std::vector<ov::Shape>> inShapes3D = {{{5, 30, 1}}};

std::vector<std::vector<ov::Shape>> inShapes4D = {
        {{1, 10, 7, 4}},
        {{1, 2, 204, 62}},
        {{3, 20, 1, 15}},
        {{1, 48, 160, 80}},
};

std::vector<int64_t> axis3D = {0, 1, 2};
std::vector<int64_t> axis4D = {0, 1, 2, 3};

const auto params2D = testing::Combine(testing::ValuesIn(modelType),
                                       testing::ValuesIn(static_shapes_to_test_representation(inShapes2D)),
                                       testing::ValuesIn(axis2D), testing::Values(DEVICE_NPU));

const auto params3D = testing::Combine(testing::ValuesIn(modelType),
                                       testing::ValuesIn(static_shapes_to_test_representation(inShapes3D)),
                                       testing::ValuesIn(axis3D), testing::Values(DEVICE_NPU));

const auto params4D = testing::Combine(testing::ValuesIn(modelType),
                                       testing::ValuesIn(static_shapes_to_test_representation(inShapes4D)),
                                       testing::ValuesIn(axis4D), testing::Values(DEVICE_NPU));

const auto paramsTiling = testing::Combine(testing::ValuesIn(modelType),
                                           testing::ValuesIn({static_shapes_to_test_representation({inShapes4D[3]})}),
                                           testing::Values(1), testing::Values(DEVICE_NPU));

/* ============= LogSoftmax NPU3720 ============= */

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_2D, LogSoftmaxLayerTest_NPU3720, params2D,
                         LogSoftmaxLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_3D, LogSoftmaxLayerTest_NPU3720, params3D,
                         LogSoftmaxLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_4D, LogSoftmaxLayerTest_NPU3720, params4D,
                         LogSoftmaxLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_tiling, LogSoftmaxLayerTest_NPU3720, paramsTiling,
                         LogSoftmaxLayerTest_NPU3720::getTestCaseName);

/* ============= LogSoftmax NPU4000 ============= */

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_2D, LogSoftmaxLayerTest_NPU4000, params2D,
                         LogSoftmaxLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_LogSoftmax_3D, LogSoftmaxLayerTest_NPU4000, params3D,
                         LogSoftmaxLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_LogSoftmax_4D, LogSoftmaxLayerTest_NPU4000, params4D,
                         LogSoftmaxLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_tiling, LogSoftmaxLayerTest_NPU4000, paramsTiling,
                         LogSoftmaxLayerTest_NPU4000::getTestCaseName);

}  // namespace
