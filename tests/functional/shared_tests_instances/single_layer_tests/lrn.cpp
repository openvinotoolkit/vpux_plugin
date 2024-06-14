//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/lrn.hpp"
#include <vector>
#include "npu_private_properties.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class LrnLayerTestCommon : public LrnLayerTest, virtual public VpuOv2LayerTest {};
class LrnLayerTest_NPU3700 : public LrnLayerTestCommon {};
class LrnLayerTest_FP16_NPU3720 : public LrnLayerTestCommon {};
class LrnLayerTest_FP16_NPU4000 : public LrnLayerTestCommon {};
class LrnLayerTest_SW_FP32 : public LrnLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "convert-precision-to-fp16=false";
    }
};
class LrnLayerTest_FP32_NPU3720 : public LrnLayerTest_SW_FP32 {};
class LrnLayerTest_FP32_NPU4000 : public LrnLayerTest_SW_FP32 {};

TEST_P(LrnLayerTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

// FP16
TEST_P(LrnLayerTest_FP16_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(LrnLayerTest_FP16_NPU4000, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(LrnLayerTest_FP32_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(LrnLayerTest_FP32_NPU4000, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const double alpha = 9.9e-05;
const double beta = 2;
const double bias = 1.0;
const size_t size = 5;

//
// NPU3700
//
const std::vector<std::vector<int64_t>> axes_3700 = {{1}, {2, 3}};

INSTANTIATE_TEST_SUITE_P(smoke_LrnCheck, LrnLayerTest_NPU3700,
                         ::testing::Combine(::testing::Values(alpha), ::testing::Values(beta), ::testing::Values(bias),
                                            ::testing::Values(size), ::testing::ValuesIn(axes_3700),
                                            ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(
                                                    std::vector<std::vector<ov::Shape>>({{{1, 10, 3, 2}}}))),
                                            ::testing::Values(DEVICE_NPU)),
                         LrnLayerTest_NPU3700::getTestCaseName);

//
// NPU3720/4000
//
const std::vector<std::vector<int64_t>> axes = {{1}, {2}, {1, 2}, {2, 3}, {1, 2, 3}};

const auto lrnParams_FP16 = ::testing::Combine(
        ::testing::Values(alpha), ::testing::Values(beta), ::testing::Values(bias), ::testing::Values(size),
        ::testing::ValuesIn(axes), ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 10, 3, 2}}}))),
        ::testing::Values(DEVICE_NPU));

const auto lrnGooglenetV1Params_FP16 = ::testing::Combine(
        ::testing::Values(9.9e-05),                    // alpha
        ::testing::Values(0.75),                       // beta
        ::testing::Values(1.0),                        // bias
        ::testing::Values(5),                          // size
        ::testing::Values(std::vector<int64_t>({1})),  // axes
        ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{1, 64, 56, 56}}}))),
        ::testing::Values(DEVICE_NPU));

const auto lrnParams_FP32 = ::testing::Combine(::testing::Values(9.9e-05),                    // alpha
                                               ::testing::Values(0.75),                       // beta
                                               ::testing::Values(1.0),                        // bias
                                               ::testing::Values(5),                          // size
                                               ::testing::Values(std::vector<int64_t>({1})),  // axes
                                               ::testing::Values(ov::element::f32),
                                               ::testing::ValuesIn(static_shapes_to_test_representation(
                                                       std::vector<std::vector<ov::Shape>>({{{1, 32, 56, 56}}}))),
                                               ::testing::Values(DEVICE_NPU));

//
// NPU3720 Instantiation
// FP16 HW

INSTANTIATE_TEST_SUITE_P(smoke_precommit_LrnCheck, LrnLayerTest_FP16_NPU3720, lrnParams_FP16,
                         LrnLayerTest_FP16_NPU3720::getTestCaseName);

//
// NPU4000 Instantiation
// FP16 SW

INSTANTIATE_TEST_SUITE_P(smoke_precommit_LrnGooglenetV1, LrnLayerTest_FP16_NPU4000, lrnGooglenetV1Params_FP16,
                         LrnLayerTest_FP16_NPU4000::getTestCaseName);

//
// NPU3720/4000 Instantiation
// FP32 SW

INSTANTIATE_TEST_SUITE_P(smoke_precommit_LrnGooglenetV1_FP32, LrnLayerTest_FP32_NPU3720, lrnParams_FP32,
                         LrnLayerTest_FP32_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_LrnGooglenetV1_FP32, LrnLayerTest_FP32_NPU4000, lrnParams_FP32,
                         LrnLayerTest_FP32_NPU4000::getTestCaseName);

}  // namespace
