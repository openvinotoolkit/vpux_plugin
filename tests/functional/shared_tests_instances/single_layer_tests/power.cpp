//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/power.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class PowerLayerTest_NPU3700 : public PowerLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(PowerLayerTest_NPU3700, HW) {
    setSkipCompilationCallback([this](std::stringstream& skip) {
        if (envConfig.IE_NPU_TESTS_RUN_INFER) {
            skip << "layer test networks hang the board";
        }
    });
    setSkipInferenceCallback([](std::stringstream& skip) {
        skip << "comparison fails";
    });
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}
}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

std::vector<std::vector<ov::Shape>> inShapes = {{{1, 8}},   {{2, 16}},  {{3, 32}},  {{4, 64}},
                                                {{5, 128}}, {{6, 256}}, {{7, 512}}, {{8, 1024}}};

std::vector<std::vector<float>> Power = {
        {0.0f}, {0.5f}, {1.0f}, {1.1f}, {1.5f}, {2.0f},
};

std::vector<ov::element::Type> modelTypes = {ov::element::f16};

// Tracking number [E#85137]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_power, PowerLayerTest_NPU3700,
                         ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShapes)),
                                            ::testing::ValuesIn(modelTypes), ::testing::ValuesIn(Power),
                                            ::testing::Values(DEVICE_NPU)),
                         PowerLayerTest_NPU3700::getTestCaseName);

}  // namespace
