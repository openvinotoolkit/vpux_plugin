// Copyright (C) 2019-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_op_tests/convolution_backprop_data.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class ConvolutionBackpropDataLayerTestCommon :
        public ConvolutionBackpropDataLayerTest,
        virtual public VpuOv2LayerTest {};
class ConvolutionBackpropDataLayerTest_NPU3700 : public ConvolutionBackpropDataLayerTestCommon {};
class ConvolutionBackpropDataLayerTest_NPU3720 : public ConvolutionBackpropDataLayerTestCommon {};

TEST_P(ConvolutionBackpropDataLayerTest_NPU3720, HW) {
    abs_threshold = 0.5;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ConvolutionBackpropDataLayerTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> netPrecisions = {ov::element::f16};

const std::vector<size_t> numOutChannels = {16};
const std::vector<size_t> specificNumOutChannels = {128};
const std::vector<ov::Shape> emptyOutputShape = {{}};
const std::vector<ov::Shape> outputShape = {{32, 64}};
const std::vector<std::vector<ptrdiff_t>> emptyOutputPadding = {{}};

/* ============= 1D ConvolutionBackpropData ============= */
const std::vector<std::vector<ov::Shape>> inputShapes1D = {{{1, 3, 30}}};
const std::vector<std::vector<size_t>> kernels1D = {{2}};
const std::vector<std::vector<size_t>> strides1D = {{2}};
const std::vector<std::vector<ptrdiff_t>> padBegins1D = {{0}};
const std::vector<std::vector<ptrdiff_t>> padEnds1D = {{0}};
const std::vector<std::vector<size_t>> dilations1D = {{1}};

const auto conv1DParams_AutoPadValid = ::testing::Combine(::testing::ValuesIn(kernels1D),       // Kernel size
                                                          ::testing::ValuesIn(strides1D),       // Strides
                                                          ::testing::ValuesIn(padBegins1D),     // Pad begin
                                                          ::testing::ValuesIn(padEnds1D),       // Pad end
                                                          ::testing::ValuesIn(dilations1D),     // Dilation
                                                          ::testing::ValuesIn(numOutChannels),  // Num out channels
                                                          ::testing::Values(ov::op::PadType::VALID),  // Padding type
                                                          ::testing::ValuesIn(emptyOutputPadding));   // Output padding
const auto conv1DParams_AutoPadValidCases =
        ::testing::Combine(conv1DParams_AutoPadValid,
                           ::testing::ValuesIn(netPrecisions),                                        // Net precision
                           ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes1D)),  // Input shapes
                           ::testing::ValuesIn(emptyOutputShape),                                     // Output shapes
                           ::testing::Values(DEVICE_NPU));                                            // Device name

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ConvolutionBackpropData1D_TestConv1DToConv2D,
                         ConvolutionBackpropDataLayerTest_NPU3720, conv1DParams_AutoPadValidCases,
                         ConvolutionBackpropDataLayerTest_NPU3720::getTestCaseName);

/* ============= 2D ConvolutionBackpropData ============= */
const std::vector<std::vector<ov::Shape>> inputShapes2D = {{{1, 16, 30, 30}}};
const std::vector<std::vector<ov::Shape>> inputShapes2D_MLIR = {
        {{1, 3, 30, 30}}, {{1, 32, 23, 30}}, {{1, 32, 46, 60}}, {{1, 32, 92, 120}}, {{1, 32, 184, 240}}};
const std::vector<std::vector<ov::Shape>> specificInputShapes2D_MLIR = {{{1, 256, 16, 32}}};

const std::vector<std::vector<size_t>> kernels2D = {{2, 2}};
const std::vector<std::vector<size_t>> specificKernels2D = {{4, 4}};
const std::vector<std::vector<size_t>> strides2D = {{2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> outputPadding2D = {{1, 1}};
const std::vector<std::vector<size_t>> dilations2D = {{1, 1}};

const auto conv2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels2D), ::testing::ValuesIn(strides2D), ::testing::ValuesIn(padBegins2D),
        ::testing::ValuesIn(padEnds2D), ::testing::ValuesIn(dilations2D), ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ov::op::PadType::EXPLICIT), ::testing::ValuesIn(emptyOutputPadding));
const auto conv2DParams_OutputPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels2D), ::testing::ValuesIn(strides2D), ::testing::ValuesIn(padBegins2D),
        ::testing::ValuesIn(padEnds2D), ::testing::ValuesIn(dilations2D), ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ov::op::PadType::EXPLICIT), ::testing::ValuesIn(outputPadding2D));
const auto conv2DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels2D), ::testing::ValuesIn(strides2D),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})), ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(dilations2D), ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ov::op::PadType::VALID), ::testing::ValuesIn(emptyOutputPadding));
const auto conv2DParams_AutoPadSameLower = ::testing::Combine(
        ::testing::ValuesIn(specificKernels2D), ::testing::ValuesIn(strides2D),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})), ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(dilations2D), ::testing::ValuesIn(specificNumOutChannels),
        ::testing::Values(ov::op::PadType::SAME_LOWER), ::testing::ValuesIn(emptyOutputPadding));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ConvolutionBackpropData2D_OutputPadding,
                         ConvolutionBackpropDataLayerTest_NPU3720,
                         ::testing::Combine(conv2DParams_OutputPadding, ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes2D)),
                                            ::testing::ValuesIn(emptyOutputShape), ::testing::Values(DEVICE_NPU)),
                         ConvolutionBackpropDataLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_ConvolutionBackpropData2D_ExplicitPadding, ConvolutionBackpropDataLayerTest_NPU3700,
        ::testing::Combine(conv2DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes2D_MLIR)),
                           ::testing::ValuesIn(emptyOutputShape), ::testing::Values(DEVICE_NPU)),
        ConvolutionBackpropDataLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_ConvolutionBackpropData2D_OutputShape, ConvolutionBackpropDataLayerTest_NPU3700,
        ::testing::Combine(conv2DParams_AutoPadSameLower, ::testing::ValuesIn(netPrecisions),
                           ::testing::ValuesIn(static_shapes_to_test_representation(specificInputShapes2D_MLIR)),
                           ::testing::ValuesIn(outputShape), ::testing::Values(DEVICE_NPU)),
        ConvolutionBackpropDataLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_ConvolutionBackpropData2D_AutoPadValid, ConvolutionBackpropDataLayerTest_NPU3700,
        ::testing::Combine(conv2DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes2D_MLIR)),
                           ::testing::ValuesIn(emptyOutputShape), ::testing::Values(DEVICE_NPU)),
        ConvolutionBackpropDataLayerTest_NPU3700::getTestCaseName);

/* ============= 3D ConvolutionBackpropData ============= */
const std::vector<std::vector<ov::Shape>> inputShapes3D = {
        {{1, 3, 10, 10, 10}}, {{1, 16, 5, 5, 5}}, {{1, 32, 5, 5, 5}}};
const std::vector<std::vector<size_t>> kernels3D = {{1, 1, 1}, {3, 3, 3}};
const std::vector<std::vector<size_t>> strides3D = {{1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins3D = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds3D = {{0, 0, 0}, {1, 1, 1}};
const std::vector<std::vector<size_t>> dilations3D = {{1, 1, 1}, {2, 2, 2}};

const auto conv3DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels3D), ::testing::ValuesIn(strides3D), ::testing::ValuesIn(padBegins3D),
        ::testing::ValuesIn(padEnds3D), ::testing::ValuesIn(dilations3D), ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ov::op::PadType::EXPLICIT), ::testing::ValuesIn(emptyOutputPadding));
const auto conv3DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels3D), ::testing::ValuesIn(strides3D),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::ValuesIn(dilations3D), ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ov::op::PadType::VALID), ::testing::ValuesIn(emptyOutputPadding));

// Tracking number [E#85137]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_ConvolutionBackpropData3D_ExplicitPadding,
                         ConvolutionBackpropDataLayerTest_NPU3700,
                         ::testing::Combine(conv3DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes3D)),
                                            ::testing::ValuesIn(emptyOutputShape), ::testing::Values(DEVICE_NPU)),
                         ConvolutionBackpropDataLayerTest_NPU3700::getTestCaseName);

// Tracking number [E#85137]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_ConvolutionBackpropData3D_AutoPadValid,
                         ConvolutionBackpropDataLayerTest_NPU3700,
                         ::testing::Combine(conv3DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes3D)),
                                            ::testing::ValuesIn(emptyOutputShape), ::testing::Values(DEVICE_NPU)),
                         ConvolutionBackpropDataLayerTest_NPU3700::getTestCaseName);

}  // namespace
