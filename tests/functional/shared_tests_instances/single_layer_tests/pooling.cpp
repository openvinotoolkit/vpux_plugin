// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/pooling.hpp"
#include "npu_private_properties.hpp"

#include <vector>

#include <common/functions.h>
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

// Option added base on CI request to decrease test runtime
// Important to enable macro (remove //) to run full tests in CI every time your change can impact AVG/MAX pool.
// Both operations are transformed in some scenario to NCE task, so it is important to enable testing when touch any of
// this mlir passes.
// #define ENABLE_ALL_POOL_TESTS

#ifdef ENABLE_ALL_POOL_TESTS
#define INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(A, B, C, D) INSTANTIATE_TEST_SUITE_P(A, B, C, D)
#else
#define INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(A, B, C, D) INSTANTIATE_TEST_SUITE_P(DISABLED_##A, B, C, D)
#endif

class PoolingLayerTest_NPU3700 : public PoolingLayerTest, virtual public VpuOv2LayerTest {};

void skipCompilationCallbackImplNPU3700(std::stringstream& skip, ov::AnyMap configuration, PoolingTypes poolType,
                                        std::vector<size_t> strides, double rel_threshold) {
    if (poolType == PoolingTypes::AVG && configuration[ov::intel_npu::compilation_mode.name()] == "DefaultHW") {
        rel_threshold = 0.25;
    }

    // MLIR uses software layer, which seem to be flawed
    if (poolType == PoolingTypes::AVG) {
        if (strides[0] != 1 || strides[1] != 1) {
            skip << "AVG pool strides != 1 produces inaccurate results";
        }
    }
}
void skipCompilationCallbackImplNPU3720(std::stringstream& skip, ov::AnyMap configuration, PoolingTypes poolType,
                                        std::vector<size_t> kernel, std::vector<size_t> strides,
                                        std::vector<size_t> padBegin, std::vector<size_t> padEnd,
                                        ov::op::PadType padType, bool excludePad, ov::element::Type netPrecision,
                                        ov::op::RoundingType roundingMode, ov::Shape inputShapes) {
    // all DefaultHW test Should be enable when E#94485 will be fixed. Convert to HW scenario not implemented
    if ((poolType == PoolingTypes::AVG) && (configuration[ov::intel_npu::compilation_mode.name()] == "DefaultHW") &&
        (strides.size() == 2)) {
        // support exclude pad for reduce number of scenario, when HandleExcludePadForAvgPoolPass fail,
        // excludePad remain, should not validate ConvertIEToVPUNCEPass for AvgPool
        if (excludePad) {
            std::vector<size_t> ones{1, 1};
            if ((padBegin != ones) || (padEnd != ones) || (strides != ones)) {
                skip << "AVGPool convert to NCE with excludePad, invalid conversion";
            }
        }
        // special implementation in reference for CEIL rounding mode with padBegin=0, padEnd=0, and Ceil rounding
        // involve padding with 1. see openvino reference:
        // openvino/src/core/reference/include/ngraph/runtime/reference/avg_pool.hpp
        // if all are 0, and CEIL request in fact padding, then enable excludePad, else, if just 1 of
        // value are not 0, work as expected, divide by constant kernel size.
        // Hw should implement in same way, or allow go to SW implementation.
        if (roundingMode == ov::op::RoundingType::CEIL) {
            std::vector<size_t> zeros{0, 0};
            if ((padBegin == zeros) && (padEnd == zeros)) {
                skip << "AVG pool CEIL rounding with PADS 0 are not proper converted to NCE AvgPool";
            }
        }
        // Default HW pipe produce wrong values for this combination, if input size is just 8x8 that produce 3x3 or
        // 1x1 output, SW reference pipeline pass. Probably align issue in HW version. Padding mode is valid
        if ((inputShapes[3] <= 8) && (inputShapes[2] <= 8) && (strides[0] == 2) && (strides[1] == 2) &&
            (padType == ov::op::PadType::VALID)) {
            skip << "AVG Pool VALID pad type for small resolution invalid conversion to NCE AvgPool";
        }
    }

    // Invalid padding with 0 for MaxPool, should be -MaxFloat
    // src/vpux_compiler/src/dialect/IE/passes/handle_large_pads.cpp pad with 0 as for Avg, but Max is not the same.
    // Remove when  E#99182 will be fixed. Or open a separate ticket related to E#69906
    if ((poolType == PoolingTypes::MAX) && (configuration[ov::intel_npu::compilation_mode.name()] == "DefaultHW") &&
        (strides.size() == 2)) {
        size_t kernel0 = kernel[0];
        size_t kernel1 = kernel[1];
        if (roundingMode == ov::op::RoundingType::CEIL) {
            kernel0 -= 1;
            kernel1 -= 1;
        }
        if ((padBegin[0] >= kernel0) || (padBegin[1] >= kernel1) || (padEnd[0] >= kernel0) || (padEnd[1] >= kernel1)) {
            skip << "MAX pool, Hw NCE version produce invalid 0 values on pad area";
        }
    }

    // MaxPool Opset1 and AvgPool Opset1 does not support i8 or u8 dataType at OV evaluator level, but works in networks
    if (((poolType == PoolingTypes::MAX) || poolType == (PoolingTypes::AVG)) &&
        (netPrecision == ov::element::i8 || netPrecision == ov::element::u8)) {
        skip << "Max pool SingleLayerTest is not enabled with precision: " << netPrecision;
    }
}

TEST_P(PoolingLayerTest_NPU3700, SW) {
    setSkipCompilationCallback([this](std::stringstream& skip) {
        const auto& poolParams = std::get<0>(GetParam());
        PoolingTypes poolType = std::get<0>(poolParams);
        std::vector<size_t> strides = std::get<2>(poolParams);
        ov::op::RoundingType roundingMode = std::get<5>(poolParams);
        skipCompilationCallbackImplNPU3700(skip, configuration, poolType, strides, rel_threshold);
    });
    setReferenceSoftwareMode();
    run(Platform::NPU3700);
}

TEST_P(PoolingLayerTest_NPU3700, HW) {
    setSkipCompilationCallback([this](std::stringstream& skip) {
        const auto& poolParams = std::get<0>(GetParam());
        PoolingTypes poolType = std::get<0>(poolParams);
        std::vector<size_t> strides = std::get<2>(poolParams);
        ov::op::RoundingType roundingMode = std::get<5>(poolParams);
        skipCompilationCallbackImplNPU3700(skip, configuration, poolType, strides, rel_threshold);
    });
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

class PoolingLayerTest_NPU3720 : public PoolingLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(PoolingLayerTest_NPU3720, SW) {
    setSkipCompilationCallback([this](std::stringstream& skip) {
        std::vector<InputShape> inputShapesOrig = std::get<2>(GetParam());
        ov::Shape inputShapes = inputShapesOrig[0].second[0];
        const auto& poolParams = std::get<0>(GetParam());
        PoolingTypes poolType;
        std::vector<size_t> kernel;
        std::vector<size_t> strides;
        std::vector<size_t> padBegin;
        std::vector<size_t> padEnd;
        ov::op::RoundingType roundingMode;
        ov::op::PadType padType;
        bool excludePad;
        const auto netPrecision = std::get<1>(GetParam());
        std::tie(poolType, kernel, strides, padBegin, padEnd, roundingMode, padType, excludePad) = poolParams;
        skipCompilationCallbackImplNPU3720(skip, configuration, poolType, kernel, strides, padBegin, padEnd, padType,
                                           excludePad, netPrecision, roundingMode, inputShapes);
    });
    abs_threshold = 0.02;
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(PoolingLayerTest_NPU3720, HW) {
    setSkipCompilationCallback([this](std::stringstream& skip) {
        std::vector<InputShape> inputShapesOrig = std::get<2>(GetParam());
        ov::Shape inputShapes = inputShapesOrig[0].second[0];
        const auto& poolParams = std::get<0>(GetParam());
        PoolingTypes poolType;
        std::vector<size_t> kernel;
        std::vector<size_t> strides;
        std::vector<size_t> padBegin;
        std::vector<size_t> padEnd;
        ov::op::RoundingType roundingMode;
        ov::op::PadType padType;
        bool excludePad;
        const auto netPrecision = std::get<1>(GetParam());
        std::tie(poolType, kernel, strides, padBegin, padEnd, roundingMode, padType, excludePad) = poolParams;
        skipCompilationCallbackImplNPU3720(skip, configuration, poolType, kernel, strides, padBegin, padEnd, padType,
                                           excludePad, netPrecision, roundingMode, inputShapes);
    });
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

using PoolingLayerTest_NPU3720_SingleCluster = PoolingLayerTest_NPU3720;

TEST_P(PoolingLayerTest_NPU3720_SingleCluster, HW) {
    setSkipCompilationCallback([this](std::stringstream& skip) {
        std::vector<InputShape> inputShapesOrig = std::get<2>(GetParam());
        ov::Shape inputShapes = inputShapesOrig[0].second[0];
        const auto& poolParams = std::get<0>(GetParam());
        PoolingTypes poolType;
        std::vector<size_t> kernel;
        std::vector<size_t> strides;
        std::vector<size_t> padBegin;
        std::vector<size_t> padEnd;
        ov::op::RoundingType roundingMode;
        ov::op::PadType padType;
        bool excludePad;
        const auto netPrecision = std::get<1>(GetParam());
        std::tie(poolType, kernel, strides, padBegin, padEnd, roundingMode, padType, excludePad) = poolParams;
        skipCompilationCallbackImplNPU3720(skip, configuration, poolType, kernel, strides, padBegin, padEnd, padType,
                                           excludePad, netPrecision, roundingMode, inputShapes);
    });
    setDefaultHardwareMode();
    setSingleClusterMode();
    useELFCompilerBackend();
    run(Platform::NPU3720);
}

class PoolingLayerTest_NPU4000 : public PoolingLayerTest, virtual public VpuOv2LayerTest {};

class PoolingLayerTest_NPU4000_F32 : public PoolingLayerTest_NPU4000 {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "convert-precision-to-fp16=false";
    }
};

TEST_P(PoolingLayerTest_NPU4000, SW) {
    abs_threshold = 0.02;
    setSkipCompilationCallback([this](std::stringstream& skip) {
        const auto& poolParams = std::get<0>(GetParam());
        PoolingTypes poolType = std::get<0>(poolParams);
        const auto netPrecision = std::get<1>(GetParam());
        if (((poolType == PoolingTypes::MAX) || poolType == (PoolingTypes::AVG)) &&
            (netPrecision == ov::element::i8 || netPrecision == ov::element::u8)) {
            skip << "Max pool SingleLayerTest is not enabled with precision: " << netPrecision;
        }
    });
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

TEST_P(PoolingLayerTest_NPU4000_F32, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

/* ============= AutoPadValid ============= */

std::vector<std::vector<ov::Shape>> inputShapesAutoPadValid = {
        {{1, 8, 32, 32}}, {{1, 16, 24, 24}}, {{1, 24, 16, 16}}, {{1, 32, 8, 8}}};
const auto pool_AutoPadValid = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),     //
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 3}, {5, 5}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}, {2, 2}}),  // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),          // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),          // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),              //
                           ::testing::Values(ov::op::PadType::VALID),                   //
                           ::testing::Values(false)),  // excludePad,                          //
        ::testing::Values(ov::element::f16),           // netPrc
        ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesAutoPadValid)),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_AutoPadValid, PoolingLayerTest_NPU3700, pool_AutoPadValid,
                         PoolingLayerTest::getTestCaseName);

/* ============= ExplicitPadding ============= */

std::vector<std::vector<ov::Shape>> inputShapesExplicitPadding = {{{1, 16, 30, 30}}};
const auto pool_ExplicitPadding = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),             //
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),                  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{2, 2}}),                  // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}, {1, 1}, {0, 1}}),  // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}, {1, 1}, {0, 1}}),  // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR,
                                             ov::op::RoundingType::CEIL),                       //
                           ::testing::Values(ov::op::PadType::EXPLICIT),                        //
                           ::testing::Values(false)),                                           //
        ::testing::Values(ov::element::f16),                                                    // netPrc
        ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesExplicitPadding)),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_ExplicitPadding, PoolingLayerTest_NPU3700, pool_ExplicitPadding,
                         PoolingLayerTest::getTestCaseName);

/* ============= AsymmetricKernel ============= */

const auto pool_AsymmetricKernel = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),             //
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 1}, {1, 3}}),          // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}, {2, 2}}),          // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                  // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                  // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                      //
                           ::testing::Values(ov::op::PadType::VALID),                           //
                           ::testing::Values(false)),                                           // excludePad
        ::testing::Values(ov::element::f16),                                                    // netPrc
        ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesExplicitPadding)),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_AsymmetricKernel, PoolingLayerTest_NPU3700, pool_AsymmetricKernel,
                         PoolingLayerTest::getTestCaseName);

/* ============= AsymmetricStrides ============= */

const auto pool_AsymmetricStrides = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),             //
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),                  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 2}, {2, 1}}),          // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                  // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                  // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                      //
                           ::testing::Values(ov::op::PadType::VALID),                           //
                           ::testing::Values(false)),                                           // excludePad
        ::testing::Values(ov::element::f16),                                                    // netPrc
        ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesExplicitPadding)),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_AsymmetricStrides, PoolingLayerTest_NPU3700, pool_AsymmetricStrides,
                         PoolingLayerTest::getTestCaseName);

/* ============= LargeSize ============= */

const auto pool_LargeSize1 = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                //
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{2, 2}}),  // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),  // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),      //
                           ::testing::Values(ov::op::PadType::VALID),           //
                           ::testing::Values(false)),                           // excludePad, //
        ::testing::Values(ov::element::f16),                                    // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 64, 128, 128}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargeSize1, PoolingLayerTest_NPU3700, pool_LargeSize1,
                         PoolingLayerTest::getTestCaseName);

const auto pool_LargeSize2 = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                //
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{2, 2}}),  // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),  // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),      //
                           ::testing::Values(ov::op::PadType::VALID),           //
                           ::testing::Values(false)),                           // excludePad
        ::testing::Values(ov::element::f16),                                    // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 256, 256}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargeSize2, PoolingLayerTest_NPU3700, pool_LargeSize2,
                         PoolingLayerTest::getTestCaseName);

/* ============= LargeStrides ============= */

const auto pool_LargeStrides = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                          //
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 3}, {11, 11}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{9, 9}}),            // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),            // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),            // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                //
                           ::testing::Values(ov::op::PadType::VALID),                     //
                           ::testing::Values(false)),                                     // excludePad

        ::testing::Values(ov::element::f16),  // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 64, 64}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargeStrides, PoolingLayerTest_NPU3700, pool_LargeStrides,
                         PoolingLayerTest::getTestCaseName);

/* ============= BatchN to batch1 ============= */

const auto pool_batchN = ::testing::Combine(::testing::Values(PoolingTypes::MAX),                //
                                            ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),  // kernels
                                            ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),  // strides
                                            ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),  // padBegins
                                            ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),  // padEnds
                                            ::testing::Values(ov::op::RoundingType::FLOOR),      //
                                            ::testing::Values(ov::op::PadType::VALID),           //
                                            ::testing::Values(false)                             // excludePad
);

INSTANTIATE_TEST_CASE_P(smoke_Pooling_BatchN, PoolingLayerTest_NPU3700,
                        ::testing::Combine(pool_batchN,                          //
                                           ::testing::Values(ov::element::f16),  // netPrc
                                           ::testing::Values(static_shapes_to_test_representation(
                                                   std::vector<ov::Shape>{{16, 16, 1, 64}})),  // inputShapes
                                           ::testing::Values(DEVICE_NPU)),
                        PoolingLayerTest::getTestCaseName);

/* ============= Padding valitation ( > K_SZ/2) ============= */

const auto pool_LargePadding2 = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                        //
                           ::testing::ValuesIn<std::vector<size_t>>({{2, 2}, {3, 3}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),          // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{2, 2}}),          // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{2, 2}}),          // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),              //
                           ::testing::Values(ov::op::PadType::VALID),                   //
                           ::testing::Values(false)),                                   // excludePad
        ::testing::Values(ov::element::f16),                                            // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 64, 64}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargePadding2, PoolingLayerTest_NPU3700, pool_LargePadding2,
                         PoolingLayerTest::getTestCaseName);

const auto pool_LargePadding3 = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                                //
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 3}, {4, 4}, {5, 5}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),                  // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),                  // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),                  // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                      //
                           ::testing::Values(ov::op::PadType::VALID),                           //
                           ::testing::Values(false)),                                           // excludePad
        ::testing::Values(ov::element::f16),                                                    // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 64, 64}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargePadding3, PoolingLayerTest_NPU3700, pool_LargePadding3,
                         PoolingLayerTest::getTestCaseName);

const auto pool_LargePadding4 = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                                        //
                           ::testing::ValuesIn<std::vector<size_t>>({{4, 4}, {5, 5}, {6, 6}, {7, 7}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),                          // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{4, 4}}),                          // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{4, 4}}),                          // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                              //
                           ::testing::Values(ov::op::PadType::VALID),                                   //
                           ::testing::Values(false)),                                                   // excludePad
        ::testing::Values(ov::element::f16),                                                            // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 64, 64}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargePadding4, PoolingLayerTest_NPU3700, pool_LargePadding4,
                         PoolingLayerTest::getTestCaseName);

const auto pool_LargePadding5 = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(PoolingTypes::MAX),                                                //
                ::testing::ValuesIn<std::vector<size_t>>({{5, 5}, {6, 6}, {7, 7}, {8, 8}, {9, 9}}),  // kernels
                ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),                                  // strides
                ::testing::ValuesIn<std::vector<size_t>>({{5, 5}}),                                  // padBegins
                ::testing::ValuesIn<std::vector<size_t>>({{5, 5}}),                                  // padEnds
                ::testing::Values(ov::op::RoundingType::FLOOR),                                      //
                ::testing::Values(ov::op::PadType::VALID),                                           //
                ::testing::Values(false)),                                                           // excludePad
        ::testing::Values(ov::element::f16),                                                         // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 64, 64}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargePadding5, PoolingLayerTest_NPU3700, pool_LargePadding5,
                         PoolingLayerTest::getTestCaseName);

const auto pool_LargePadding6 = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),  //
                           ::testing::ValuesIn<std::vector<size_t>>(
                                   {{6, 6}, {7, 7}, {8, 8}, {9, 9}, {10, 10}, {11, 11}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),             // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{6, 6}}),             // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{6, 6}}),             // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                 //
                           ::testing::Values(ov::op::PadType::VALID),                      //
                           ::testing::Values(false)),                                      // excludePad
        ::testing::Values(ov::element::f16),                                               // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 64, 64}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargePadding6, PoolingLayerTest_NPU3700, pool_LargePadding6,
                         PoolingLayerTest::getTestCaseName);

const auto pool_LargePadding7 = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(PoolingTypes::MAX),                                                    //
                ::testing::ValuesIn<std::vector<size_t>>({{7, 7}, {8, 8}, {9, 9}, {10, 10}, {11, 11}}),  // kernels
                ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),                                      // strides
                ::testing::ValuesIn<std::vector<size_t>>({{7, 7}}),                                      // padBegins
                ::testing::ValuesIn<std::vector<size_t>>({{7, 7}}),                                      // padEnds
                ::testing::Values(ov::op::RoundingType::FLOOR),                                          //
                ::testing::Values(ov::op::PadType::VALID),                                               //
                ::testing::Values(false)),                                                               // excludePad
        ::testing::Values(ov::element::f16),                                                             // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 64, 64}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargePadding7, PoolingLayerTest_NPU3700, pool_LargePadding7,
                         PoolingLayerTest::getTestCaseName);

const auto pool_LargePadding8 = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                                            //
                           ::testing::ValuesIn<std::vector<size_t>>({{8, 8}, {9, 9}, {10, 10}, {11, 11}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),                              // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{8, 8}}),                              // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{8, 8}}),                              // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                                  //
                           ::testing::Values(ov::op::PadType::VALID),                                       //
                           ::testing::Values(false)),  // excludePad
        ::testing::Values(ov::element::f16),           // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 64, 64}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargePadding8, PoolingLayerTest_NPU3700, pool_LargePadding8,
                         PoolingLayerTest::getTestCaseName);

/* ============= AVGPooling / Large Kernels ============= */

const auto avgPool_largeKernels = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG),                  //
                           ::testing::ValuesIn<std::vector<size_t>>({{23, 30}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),    // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),    // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),    // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),        //
                           ::testing::Values(ov::op::PadType::VALID),             //
                           ::testing::Values(false)),                             // excludePad, //
        ::testing::Values(ov::element::f16),                                      // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 2048, 23, 30}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_AvgPooling_LargeKernels, PoolingLayerTest_NPU3700, avgPool_largeKernels,
                         PoolingLayerTest::getTestCaseName);

/* ============= AVGPooling / Large KernelsX ============= */

const auto avgPool_largeKernelsX = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG),                                          //
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 14}}),                           // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),                            // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                            // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                            // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                                //
                           ::testing::Values(ov::op::PadType::VALID),                                     //
                           ::testing::Values(false)),                                                     // excludePad
        ::testing::Values(ov::element::f16),                                                              // netPrc
        ::testing::Values(static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 1, 14}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_AvgPooling_LargeKernelsX, PoolingLayerTest_NPU3700, avgPool_largeKernelsX,
                         PoolingLayerTest::getTestCaseName);

/* ============= AVGPooling / Large KernelsY ============= */

const auto avgPool_largeKernelsY = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG),                                          //
                           ::testing::ValuesIn<std::vector<size_t>>({{14, 1}}),                           // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),                            // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                            // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                            // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                                //
                           ::testing::Values(ov::op::PadType::VALID),                                     //
                           ::testing::Values(false)),                                                     // excludePad,
        ::testing::Values(ov::element::f16),                                                              // netPrc
        ::testing::Values(static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 14, 1}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_AvgPooling_LargeKernelsY, PoolingLayerTest_NPU3700, avgPool_largeKernelsY,
                         PoolingLayerTest::getTestCaseName);

/* ============= AVGPooling / Large Prime Kernels ============= */

const auto avgPool_largePrimeKernels = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG),                  //
                           ::testing::ValuesIn<std::vector<size_t>>({{17, 17}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),    // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),    // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),    // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),        //
                           ::testing::Values(ov::op::PadType::VALID),             //
                           ::testing::Values(false)),                             // excludePad,
        ::testing::Values(ov::element::f16),                                      // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 147, 17, 17}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_AvgPooling_LargePrimeKernels, PoolingLayerTest_NPU3700, avgPool_largePrimeKernels,
                         PoolingLayerTest::getTestCaseName);

/* ============= MAXPooling / Large Kernels ============= */

const auto maxPool_largeKernels = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                  //
                           ::testing::ValuesIn<std::vector<size_t>>({{23, 30}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{23, 30}}),  // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),    // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),    // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),        //
                           ::testing::Values(ov::op::PadType::VALID),             //
                           ::testing::Values(false)),                             // excludePad
        ::testing::Values(ov::element::f16),                                      // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 2048, 23, 30}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_MaxPooling_LargeKernels, PoolingLayerTest_NPU3700, maxPool_largeKernels,
                         PoolingLayerTest::getTestCaseName);

/* ============= MAXPooling / Large KernelsX ============= */

const auto maxPool_largeKernelsX = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                                          //
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 14}}),                           // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),                            // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                            // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                            // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                                //
                           ::testing::Values(ov::op::PadType::VALID),                                     //
                           ::testing::Values(false)),                                                     // excludePad
        ::testing::Values(ov::element::f16),                                                              // netPrc
        ::testing::Values(static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 1, 14}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_MaxPooling_LargeKernelsX, PoolingLayerTest_NPU3700, maxPool_largeKernelsX,
                         PoolingLayerTest::getTestCaseName);

/* ============= MAXPooling / Large KernelsY ============= */

const auto maxPool_largeKernelsY = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                                          //
                           ::testing::ValuesIn<std::vector<size_t>>({{14, 1}}),                           // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),                            // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                            // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),                            // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                                //
                           ::testing::Values(ov::op::PadType::VALID),                                     //
                           ::testing::Values(false)),                                                     // excludePad
        ::testing::Values(ov::element::f16),                                                              // netPrc
        ::testing::Values(static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 14, 1}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_MaxPooling_LargeKernelsY, PoolingLayerTest_NPU3700, maxPool_largeKernelsY,
                         PoolingLayerTest::getTestCaseName);

/* ============= AvgPooling / Exclude_Pad Handling ============= */

const auto avgPool_excludePad = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG),                //
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),  // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),  // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),  // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),      //
                           ::testing::Values(ov::op::PadType::VALID),           //
                           ::testing::Values(true)),                            // excludePad,
        ::testing::Values(ov::element::f16),                                    // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 16, 28, 28}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_avgPool_excludePad, PoolingLayerTest_NPU3700, avgPool_excludePad,
                         PoolingLayerTest::getTestCaseName);

/* ======================================== NPU 3720 ============================================================= */

/* ==== Custom tests scenario extra added for 3720 ===== */
const auto pool_ExplicitNoPadding_Params = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),
                           ::testing::ValuesIn<std::vector<size_t>>({{14, 14}, {14, 1}, {1, 14}}),  // kernels
                           ::testing::Values<std::vector<size_t>>({1, 1}),                          // strides
                           ::testing::Values<std::vector<size_t>>({0, 0}),                          // padBegins
                           ::testing::Values<std::vector<size_t>>({0, 0}),                          // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR), ::testing::Values(ov::op::PadType::EXPLICIT),
                           ::testing::Values(true)),  // excludePad
        ::testing::Values(ov::element::f16),          // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 30, 14, 14}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

// U-net usecase
const auto pool_unet_Params = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),
                           ::testing::Values<std::vector<size_t>>({12, 1}),  // kernels
                           ::testing::Values<std::vector<size_t>>({1, 1}),   // strides
                           ::testing::Values<std::vector<size_t>>({0, 0}),   // padBegins
                           ::testing::Values<std::vector<size_t>>({0, 0}),   // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR), ::testing::Values(ov::op::PadType::EXPLICIT),
                           ::testing::Values(true)),  // excludePad
        ::testing::Values(ov::element::f16),          // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 1, 12, 176}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

// large kernel
const auto pooling_largeKernel_Params = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),  //
                           ::testing::ValuesIn<std::vector<size_t>>({{28, 28}}),     // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),       // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),       // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),       // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),           //
                           ::testing::Values(ov::op::PadType::VALID),                //
                           ::testing::Values(true)),                                 // excludePad
        ::testing::Values(ov::element::f16),                                         // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 70, 28, 28}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

// Large kernel with stride 1
const auto pooling_largeKernelStrideOne = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                 //
                           ::testing::ValuesIn<std::vector<size_t>>({{71, 1}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1, 1}}),   // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),   // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0}}),   // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),       //
                           ::testing::Values(ov::op::PadType::VALID),            //
                           ::testing::Values(false)),                            // excludePad,              //
        ::testing::Values(ov::element::f16),                                     // netPrc
        ::testing::Values(static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 1, 71, 2}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

// Test all padding type
const auto poolAllPadTypeParams = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG, PoolingTypes::MAX),
                           ::testing::ValuesIn<std::vector<size_t>>({{5, 7}}),  // kernels
                           ::testing::Values<std::vector<size_t>>({2, 3}),      // strides
                           ::testing::Values<std::vector<size_t>>({2, 3}),      // padBegins
                           ::testing::Values<std::vector<size_t>>({1, 2}),      // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR, ov::op::RoundingType::CEIL),
                           ::testing::Values(ov::op::PadType::EXPLICIT, ov::op::PadType::SAME_LOWER,
                                             ov::op::PadType::SAME_UPPER, ov::op::PadType::VALID),
                           ::testing::Values(true)),                                                      // excludePad
        ::testing::Values(ov::element::f16),                                                              // netPrc
        ::testing::Values(static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 2, 30, 30}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

// 3D usecase
const auto pool3DParams = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG, PoolingTypes::MAX),
                           ::testing::ValuesIn<std::vector<size_t>>({{3}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{1}}),  // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{1}}),  // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0}}),  // padEnds
                           ::testing::Values(ov::op::RoundingType::CEIL),
                           ::testing::Values(ov::op::PadType::SAME_UPPER),
                           ::testing::Values(false)),                                                 // excludePad
        ::testing::Values(ov::element::f16),                                                          // netPrc
        ::testing::Values(static_shapes_to_test_representation(std::vector<ov::Shape>{{3, 4, 64}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

// 5d usecase
const auto pool5DParams = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG, PoolingTypes::MAX),
                           ::testing::ValuesIn<std::vector<size_t>>({{2, 2, 2}}),  // kernels
                           ::testing::ValuesIn<std::vector<size_t>>({{2, 2, 2}}),  // strides
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<size_t>>({{0, 0, 0}}),  // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),
                           ::testing::Values(ov::op::PadType::SAME_UPPER),
                           ::testing::Values(true)),  // excludePad
        ::testing::Values(ov::element::f16),          // netPrc
        ::testing::Values(
                static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 4, 16, 8, 12}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

// pad outside of kernel size/2. Pad is valid until at kerneSize-1.
const auto pooligBigPadEndParams = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG, PoolingTypes::MAX),
                           ::testing::ValuesIn<std::vector<size_t>>({{3, 3}}),  // kernels
                           ::testing::Values<std::vector<size_t>>({2, 2}),      // strides
                           ::testing::Values<std::vector<size_t>>({0, 0}),      // padBegins
                           ::testing::Values<std::vector<size_t>>({2, 2}),      // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR, ov::op::RoundingType::CEIL),
                           ::testing::Values(ov::op::PadType::EXPLICIT),
                           ::testing::Values(false)),                                                     // excludePad
        ::testing::Values(ov::element::f16),                                                              // netPrc
        ::testing::Values(static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 4, 54, 54}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

// basic usecase
const auto pool_basic_Params = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG, PoolingTypes::MAX),
                           ::testing::Values<std::vector<size_t>>({3, 3}),  // kernels
                           ::testing::Values<std::vector<size_t>>({1, 1}),  // strides
                           ::testing::Values<std::vector<size_t>>({1, 1}),  // padBegins
                           ::testing::Values<std::vector<size_t>>({1, 1}),  // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR), ::testing::Values(ov::op::PadType::EXPLICIT),
                           ::testing::Values(false)),                                                     // excludePad
        ::testing::Values(ov::element::f32),                                                              // netPrc
        ::testing::Values(static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 2, 16, 24}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

// Integer Input/Output
const auto pool_inputOutputInteger = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG, PoolingTypes::MAX),                      //
                           ::testing::Values<std::vector<size_t>>({{8, 8}}),                             // kernels
                           ::testing::Values<std::vector<size_t>>({{8, 1}}),                             // strides
                           ::testing::Values<std::vector<size_t>>({{0, 0}}),                             // padBegins
                           ::testing::Values<std::vector<size_t>>({{0, 0}}),                             // padEnds
                           ::testing::Values(ov::op::RoundingType::FLOOR),                               //
                           ::testing::Values(ov::op::PadType::VALID),                                    //
                           ::testing::Values(false)),                                                    // excludePad,
        ::testing::Values(ov::element::u8, ov::element::i8, ov::element::i32),                           // netPrc
        ::testing::Values(static_shapes_to_test_representation(std::vector<ov::Shape>{{1, 1, 16, 8}})),  // inputShapes
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_NCHW_NoPadding, PoolingLayerTest_NPU3720, pool_ExplicitNoPadding_Params,
                         PoolingLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_NCHW_NoPadding_ELF, PoolingLayerTest_NPU3720_SingleCluster,
                         pool_ExplicitNoPadding_Params, PoolingLayerTest_NPU3720::getTestCaseName);
// U-net usecase
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Pooling_unet, PoolingLayerTest_NPU3720, pool_unet_Params,
                         PoolingLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_unet_ELF, PoolingLayerTest_NPU3720_SingleCluster, pool_unet_Params,
                         PoolingLayerTest_NPU3720::getTestCaseName);
// large kernel
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargeKernel, PoolingLayerTest_NPU3720, pooling_largeKernel_Params,
                         PoolingLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargeKernel3720_ELF, PoolingLayerTest_NPU3720_SingleCluster,
                         pooling_largeKernel_Params, PoolingLayerTest_NPU3720::getTestCaseName);
// Large kernel with stride 1
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Pooling_LargeKernelStrideOne, PoolingLayerTest_NPU3720,
                         pooling_largeKernelStrideOne, PoolingLayerTest_NPU3720::getTestCaseName);
// all PadType
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_AllPadType, PoolingLayerTest_NPU3720, poolAllPadTypeParams,
                         PoolingLayerTest::getTestCaseName);
// 3D usecase
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_3D, PoolingLayerTest_NPU3720, pool3DParams,
                         PoolingLayerTest_NPU3720::getTestCaseName);
// 5d usecase
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_5D, PoolingLayerTest_NPU3720, pool5DParams,
                         PoolingLayerTest_NPU3720::getTestCaseName);
// pad outside of kernel size/2. Pad is valid until at kerneSize-1.
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_BigPadEndParams, PoolingLayerTest_NPU3720, pooligBigPadEndParams,
                         PoolingLayerTest_NPU3720::getTestCaseName);

// Integer Input/Output
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_InputOutputInteger, PoolingLayerTest_NPU3720, pool_inputOutputInteger,
                         PoolingLayerTest_NPU3720::getTestCaseName);

// previous cip reused tests
/* ============= AutoPadValid ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_AutoPadValid, PoolingLayerTest_NPU3720, pool_AutoPadValid,
                                             PoolingLayerTest::getTestCaseName);
/* ============= ExplicitPadding ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_ExplicitPadding, PoolingLayerTest_NPU3720,
                                             pool_ExplicitPadding, PoolingLayerTest::getTestCaseName);
/* ============= AsymmetricKernel ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_AsymmetricKernel, PoolingLayerTest_NPU3720,
                                             pool_AsymmetricKernel, PoolingLayerTest::getTestCaseName);
/* ============= AsymmetricStrides ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_AsymmetricStrides, PoolingLayerTest_NPU3720,
                                             pool_AsymmetricStrides, PoolingLayerTest::getTestCaseName);

/* ============= AVGPooling / Large Kernels ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_AvgPooling_LargeKernels, PoolingLayerTest_NPU3720,
                                             avgPool_largeKernels, PoolingLayerTest::getTestCaseName);
/* ============= AVGPooling / Large KernelsX ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_AvgPooling_LargeKernelsX, PoolingLayerTest_NPU3720,
                                             avgPool_largeKernelsX, PoolingLayerTest::getTestCaseName);
/* ============= AVGPooling / Large KernelsY ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_AvgPooling_LargeKernelsY, PoolingLayerTest_NPU3720,
                                             avgPool_largeKernelsY, PoolingLayerTest::getTestCaseName);

/* ============= AVGPooling / Large Prime Kernels ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_AvgPooling_LargePrimeKernels, PoolingLayerTest_NPU3720,
                                             avgPool_largePrimeKernels, PoolingLayerTest::getTestCaseName);
/* ============= AvgPooling / Exclude_Pad Handling ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_avgPool_excludePad, PoolingLayerTest_NPU3720, avgPool_excludePad,
                                             PoolingLayerTest::getTestCaseName);

// Max pool ported tests from VPU3700
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargeSize1, PoolingLayerTest_NPU3720, pool_LargeSize1,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargeSize2, PoolingLayerTest_NPU3720, pool_LargeSize2,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargeStrides, PoolingLayerTest_NPU3720, pool_LargeStrides,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding2, PoolingLayerTest_NPU3720, pool_LargePadding2,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding3, PoolingLayerTest_NPU3720, pool_LargePadding3,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding4, PoolingLayerTest_NPU3720, pool_LargePadding4,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding5, PoolingLayerTest_NPU3720, pool_LargePadding5,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding6, PoolingLayerTest_NPU3720, pool_LargePadding6,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding7, PoolingLayerTest_NPU3720, pool_LargePadding7,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding8, PoolingLayerTest_NPU3720, pool_LargePadding8,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_MaxPooling_LargeKernels, PoolingLayerTest_NPU3720,
                                             maxPool_largeKernels, PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_MaxPooling_LargeKernelsX, PoolingLayerTest_NPU3720,
                                             maxPool_largeKernelsX, PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_MaxPooling_LargeKernelsY, PoolingLayerTest_NPU3720,
                                             maxPool_largeKernelsY, PoolingLayerTest::getTestCaseName);

/* ======================================== NPU 4000 ============================================================= */

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_NoPadding, PoolingLayerTest_NPU4000, pool_ExplicitNoPadding_Params,
                         PoolingLayerTest::getTestCaseName);
// U-net usecase
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Pooling_unet, PoolingLayerTest_NPU4000, pool_unet_Params,
                         PoolingLayerTest::getTestCaseName);
// large kernel
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargeKernel, PoolingLayerTest_NPU4000, pooling_largeKernel_Params,
                         PoolingLayerTest::getTestCaseName);
// Large kernel with stride 1
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Pooling_LargeKernelStrideOne, PoolingLayerTest_NPU4000,
                         pooling_largeKernelStrideOne, PoolingLayerTest::getTestCaseName);
// all PadType
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_AllPadType, PoolingLayerTest_NPU4000, poolAllPadTypeParams,
                         PoolingLayerTest::getTestCaseName);
// 3D usecase
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_3D, PoolingLayerTest_NPU4000, pool3DParams, PoolingLayerTest::getTestCaseName);
// 5d usecase
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_5D, PoolingLayerTest_NPU4000, pool5DParams, PoolingLayerTest::getTestCaseName);
// pad outside of kernel size/2. Pad is valid until at kerneSize-1.
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_BigPadEndParams, PoolingLayerTest_NPU4000, pooligBigPadEndParams,
                         PoolingLayerTest_NPU4000::getTestCaseName);
// Sanity FP32 kernel check
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_F32, PoolingLayerTest_NPU4000_F32, pool_basic_Params,
                         PoolingLayerTest::getTestCaseName);
// Integer Input/Output
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_InputOutputInteger, PoolingLayerTest_NPU4000, pool_inputOutputInteger,
                         PoolingLayerTest_NPU3720::getTestCaseName);
// previous cip reused tests
/* ============= AutoPadValid ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_AutoPadValid, PoolingLayerTest_NPU4000, pool_AutoPadValid,
                                             PoolingLayerTest::getTestCaseName);
/* ============= ExplicitPadding ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_ExplicitPadding, PoolingLayerTest_NPU4000,
                                             pool_ExplicitPadding, PoolingLayerTest::getTestCaseName);
/* ============= AsymmetricKernel ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_AsymmetricKernel, PoolingLayerTest_NPU4000,
                                             pool_AsymmetricKernel, PoolingLayerTest::getTestCaseName);
/* ============= AsymmetricStrides ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_AsymmetricStrides, PoolingLayerTest_NPU4000,
                                             pool_AsymmetricStrides, PoolingLayerTest::getTestCaseName);

/* ============= AVGPooling / Large Kernels ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_AvgPooling_LargeKernels, PoolingLayerTest_NPU4000,
                                             avgPool_largeKernels, PoolingLayerTest::getTestCaseName);
/* ============= AVGPooling / Large KernelsX ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_AvgPooling_LargeKernelsX, PoolingLayerTest_NPU4000,
                                             avgPool_largeKernelsX, PoolingLayerTest::getTestCaseName);
/* ============= AVGPooling / Large KernelsY ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_AvgPooling_LargeKernelsY, PoolingLayerTest_NPU4000,
                                             avgPool_largeKernelsY, PoolingLayerTest::getTestCaseName);
/* ============= AVGPooling / Large Prime Kernels ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_AvgPooling_LargePrimeKernels, PoolingLayerTest_NPU4000,
                                             avgPool_largePrimeKernels, PoolingLayerTest::getTestCaseName);
/* ============= AvgPooling / Exclude_Pad Handling ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_avgPool_excludePad, PoolingLayerTest_NPU4000, avgPool_excludePad,
                                             PoolingLayerTest::getTestCaseName);

// Max pool ported tests from NPU3700
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargeSize1, PoolingLayerTest_NPU4000, pool_LargeSize1,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargeSize2, PoolingLayerTest_NPU4000, pool_LargeSize2,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargeStrides, PoolingLayerTest_NPU4000, pool_LargeStrides,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding2, PoolingLayerTest_NPU4000, pool_LargePadding2,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding3, PoolingLayerTest_NPU4000, pool_LargePadding3,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding4, PoolingLayerTest_NPU4000, pool_LargePadding4,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding5, PoolingLayerTest_NPU4000, pool_LargePadding5,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding6, PoolingLayerTest_NPU4000, pool_LargePadding6,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding7, PoolingLayerTest_NPU4000, pool_LargePadding7,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding8, PoolingLayerTest_NPU4000, pool_LargePadding8,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_MaxPooling_LargeKernels, PoolingLayerTest_NPU4000,
                                             maxPool_largeKernels, PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_MaxPooling_LargeKernelsX, PoolingLayerTest_NPU4000,
                                             maxPool_largeKernelsX, PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_MaxPooling_LargeKernelsY, PoolingLayerTest_NPU4000,
                                             maxPool_largeKernelsY, PoolingLayerTest::getTestCaseName);

}  // namespace
