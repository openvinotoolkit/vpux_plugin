//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/logical.hpp"

#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class LogicalLayerTestCommon : public LogicalLayerTest, virtual public VpuOv2LayerTest {};

class LogicalLayerTest_NPU3700 : public LogicalLayerTestCommon {};
class LogicalLayerTest_SW_NPU3720 : public LogicalLayerTestCommon {};
class LogicalLayerTest_HW_NPU3720 : public LogicalLayerTestCommon {};
class LogicalLayerTest_SW_NPU4000 : public LogicalLayerTestCommon {};

TEST_P(LogicalLayerTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

TEST_P(LogicalLayerTest_NPU3700, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3700);
}

TEST_P(LogicalLayerTest_SW_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(LogicalLayerTest_HW_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(LogicalLayerTest_SW_NPU4000, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

std::vector<std::vector<ov::Shape>> combineShapes(
        const std::map<ov::Shape, std::vector<ov::Shape>>& input_shapes_static) {
    std::vector<std::vector<ov::Shape>> result;
    for (const auto& input_shape : input_shapes_static) {
        for (auto& item : input_shape.second) {
            result.push_back({input_shape.first, item});
        }

        if (input_shape.second.empty()) {
            result.push_back({input_shape.first, {}});
        }
    }
    return result;
}

std::map<ov::Shape, std::vector<ov::Shape>> inputShapes = {
        {{1}, {{1}, {17}, {1, 1}, {2, 18}, {1, 1, 2}, {2, 2, 3}, {1, 1, 2, 3}}},
        {{5}, {{1}, {1, 1}, {2, 5}, {1, 1, 1}, {2, 2, 5}}},
        {{2, 200}, {{1}, {200}, {2, 2, 200}}},
        {{1, 3, 20}, {{20}, {2, 1, 1}}},
        {{2, 17, 3, 4}, {{2, 1, 3, 4}}},
};

std::map<ov::Shape, std::vector<ov::Shape>> inputShapesNot = {
        {{5}, {}},
        {{2, 200}, {}},
        {{1, 3, 20}, {}},
        {{1, 17, 3, 4}, {}},
};

std::vector<LogicalTypes> logicalOpTypes = {
        LogicalTypes::LOGICAL_AND,
        LogicalTypes::LOGICAL_OR,
        LogicalTypes::LOGICAL_XOR,
};

std::vector<InputLayerType> secondInputTypes = {
        InputLayerType::CONSTANT,
        InputLayerType::PARAMETER,
};

std::vector<ov::element::Type> modelTypes = {
        ov::element::boolean,
};

std::map<std::string, std::string> additional_config = {};

const auto LogicalTestParams = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(combineShapes(inputShapes))),  // Input shapes
        ::testing::ValuesIn(logicalOpTypes),                                                    // Logical op type
        ::testing::ValuesIn(secondInputTypes),                                                  // Second input type
        ::testing::ValuesIn(modelTypes),                                                        // Model type
        ::testing::Values(DEVICE_NPU),                                                          // Device name
        ::testing::Values(additional_config));  // Additional model configuration

const auto LogicalTestParamsNot = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(combineShapes(inputShapesNot))),
        ::testing::Values(LogicalTypes::LOGICAL_NOT), ::testing::Values(InputLayerType::CONSTANT),
        ::testing::ValuesIn(modelTypes), ::testing::Values(DEVICE_NPU), ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(smoke_CompareWithRefs, LogicalLayerTest_NPU3700, LogicalTestParams,
                        LogicalLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNot, LogicalLayerTest_NPU3700, LogicalTestParamsNot,
                         LogicalLayerTest::getTestCaseName);

//
// NPU3720/4000
//
std::set<LogicalTypes> supportedTypes = {
        LogicalTypes::LOGICAL_OR,
        LogicalTypes::LOGICAL_XOR,
        LogicalTypes::LOGICAL_AND,
};

std::map<ov::Shape, std::vector<ov::Shape>> inShapes = {
        {{2, 17, 3, 4}, {{2, 1, 3, 4}}},   {{1, 16, 32}, {{1, 16, 32}}}, {{1, 28, 300, 1}, {{1, 1, 300, 28}}},
        {{2, 17, 3, 4}, {{4}, {1, 3, 4}}}, {{2, 200}, {{2, 200}}},

};

std::map<ov::Shape, std::vector<ov::Shape>> precommit_inShapes = {
        {{1, 16, 32}, {{1, 1, 32}}},
};

std::map<ov::Shape, std::vector<ov::Shape>> inShapesNot = {
        {{1, 2, 4}, {}},
};

std::map<ov::Shape, std::vector<ov::Shape>> tiling_inShapes = {
        {{1, 10, 256, 256}, {{1, 10, 256, 256}}},
};

const auto logical_params = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(combineShapes(inShapes))),
        ::testing::ValuesIn(supportedTypes), ::testing::ValuesIn(secondInputTypes), ::testing::ValuesIn(modelTypes),
        ::testing::Values(DEVICE_NPU), ::testing::Values(additional_config));

const auto precommit_logical_params = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(combineShapes(precommit_inShapes))),
        ::testing::ValuesIn(supportedTypes), ::testing::ValuesIn(secondInputTypes), ::testing::ValuesIn(modelTypes),
        ::testing::Values(DEVICE_NPU), ::testing::Values(additional_config));

const auto precommit_logical_params_not = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(combineShapes(inShapesNot))),
        ::testing::Values(LogicalTypes::LOGICAL_NOT), ::testing::Values(InputLayerType::CONSTANT),
        ::testing::ValuesIn(modelTypes), ::testing::Values(DEVICE_NPU), ::testing::Values(additional_config));

const auto tiling_logical_params = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(combineShapes(tiling_inShapes))),
        ::testing::Values(LogicalTypes::LOGICAL_OR), ::testing::ValuesIn(secondInputTypes),
        ::testing::ValuesIn(modelTypes), ::testing::Values(DEVICE_NPU), ::testing::Values(additional_config));

// ------ NPU3720 ------
// [Tracking number E#107046]
INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_logical, LogicalLayerTest_SW_NPU3720, logical_params,
                        LogicalLayerTest::getTestCaseName);
// [Tracking number E#107046]
INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_precommit_logical, LogicalLayerTest_SW_NPU3720, precommit_logical_params,
                        LogicalLayerTest::getTestCaseName);
// [Tracking number E#107046]
INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_precommit_logical_not, LogicalLayerTest_SW_NPU3720,
                        precommit_logical_params_not, LogicalLayerTest::getTestCaseName);
// [Tracking number E#107046]
INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_tiling, LogicalLayerTest_HW_NPU3720, tiling_logical_params,
                        LogicalLayerTest::getTestCaseName);

// ------ NPU4000 ------
// [Tracking number E#107046]
INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_logical, LogicalLayerTest_SW_NPU4000, logical_params,
                        LogicalLayerTest::getTestCaseName);
// [Tracking number E#107046]
INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_precommit_logical, LogicalLayerTest_SW_NPU4000, precommit_logical_params,
                        LogicalLayerTest::getTestCaseName);
// [Tracking number E#107046]
INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_precommit_logical_not, LogicalLayerTest_SW_NPU4000,
                        precommit_logical_params_not, LogicalLayerTest::getTestCaseName);

}  // namespace
