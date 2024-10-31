//
// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/logical.hpp"

#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class LogicalLayerTestCommon : public LogicalLayerTest, virtual public VpuOv2LayerTest {};
class LogicalLayerTestHW : public LogicalLayerTestCommon {};

TEST_P(LogicalLayerTestCommon, NPU3720_SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(LogicalLayerTestHW, NPU3720) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(LogicalLayerTestCommon, NPU4000_SW) {
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

// [Tracking number E#109588]
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_logical, LogicalLayerTestCommon, logical_params,
                         LogicalLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_precommit_logical, LogicalLayerTestCommon, precommit_logical_params,
                         LogicalLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_precommit_logical_not, LogicalLayerTestCommon, precommit_logical_params_not,
                         LogicalLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_tiling, LogicalLayerTestHW, tiling_logical_params,
                         LogicalLayerTest::getTestCaseName);

}  // namespace
