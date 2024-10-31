// Copyright (C) 2019-2024 Intel Corporation
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
class ConvolutionBackpropDataLayerTest_NPU3720 : public ConvolutionBackpropDataLayerTestCommon {};

class ConvolutionBackpropDataSEPLayerTest_NPU3720 : public ConvolutionBackpropDataLayerTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] = "enable-se-ptrs-operations=true";
    }
};

class ConvolutionBackpropDataSEPLayerTest_NPU4000 : public ConvolutionBackpropDataLayerTestCommon {};

TEST_P(ConvolutionBackpropDataSEPLayerTest_NPU3720, HW) {
    rel_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ConvolutionBackpropDataSEPLayerTest_NPU4000, HW) {
    rel_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

TEST_P(ConvolutionBackpropDataLayerTest_NPU3720, HW) {
    rel_threshold = 0.01;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
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

/* ============= 2D ConvolutionBackpropData With OutputShape ============= */
const std::vector<std::vector<ov::Shape>> inputShapes2DWithOS = {{{1, 32, 128, 128}}};
const std::vector<ov::Shape> specifiedOutputShape = {{128, 128}};

const std::vector<std::vector<size_t>> kernels2DWithOS = {{2, 2}};
const std::vector<std::vector<size_t>> strides2DWithOS = {{2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins2DWithOS = {{64, 64}};
const std::vector<std::vector<ptrdiff_t>> padEnds2DWithOS = {{64, 64}};
const std::vector<std::vector<size_t>> dilations2DWithOS = {{1, 1}};

const auto conv2DParamsWithOS_ExplicitPadding =
        ::testing::Combine(::testing::ValuesIn(kernels2DWithOS), ::testing::ValuesIn(strides2DWithOS),
                           ::testing::ValuesIn(padBegins2DWithOS), ::testing::ValuesIn(padEnds2DWithOS),
                           ::testing::ValuesIn(dilations2DWithOS), ::testing::ValuesIn(numOutChannels),
                           ::testing::Values(ov::op::PadType::EXPLICIT), ::testing::ValuesIn(emptyOutputPadding));

INSTANTIATE_TEST_SUITE_P(
        smoke_precommit_ConvolutionBackpropData2DWithOutputShape_ExplicitPadding,
        ConvolutionBackpropDataLayerTest_NPU3720,
        ::testing::Combine(conv2DParamsWithOS_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
                           ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes2DWithOS)),
                           ::testing::ValuesIn(specifiedOutputShape), ::testing::Values(DEVICE_NPU)),
        ConvolutionBackpropDataLayerTest_NPU3720::getTestCaseName);

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

/* ============= 2D ConvolutionBackpropData Convert to SEP Op ============= */
const std::vector<std::vector<ov::Shape>> seInputShapes = {{{1, 16, 128, 128}}};

const std::vector<std::vector<size_t>> seKernels = {{5, 5}};
const std::vector<std::vector<size_t>> seStrides = {{2, 3}};
const std::vector<std::vector<ptrdiff_t>> sePadBegins = {{3, 1}, {2, 4}};
const std::vector<std::vector<ptrdiff_t>> sePadEnds = {{1, 3}, {4, 2}};
const std::vector<std::vector<ptrdiff_t>> seOutputPadding = {{2, 1}};
const std::vector<std::vector<size_t>> seDilations = {{1, 1}};

const auto se_conv2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(seKernels), ::testing::ValuesIn(seStrides), ::testing::ValuesIn(sePadBegins),
        ::testing::ValuesIn(sePadEnds), ::testing::ValuesIn(seDilations), ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ov::op::PadType::EXPLICIT), ::testing::ValuesIn(emptyOutputPadding));
const auto se_conv2DParams_OutputPadding = ::testing::Combine(
        ::testing::ValuesIn(seKernels), ::testing::ValuesIn(seStrides), ::testing::ValuesIn(sePadBegins),
        ::testing::ValuesIn(sePadEnds), ::testing::ValuesIn(seDilations), ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ov::op::PadType::EXPLICIT), ::testing::ValuesIn(seOutputPadding));

// ------ NPU3720 ------
INSTANTIATE_TEST_SUITE_P(smoke_precommit_SEP_ConvolutionBackpropData2D_ExplicitPadding,
                         ConvolutionBackpropDataSEPLayerTest_NPU3720,
                         ::testing::Combine(se_conv2DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(seInputShapes)),
                                            ::testing::ValuesIn(emptyOutputShape), ::testing::Values(DEVICE_NPU)),
                         ConvolutionBackpropDataSEPLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_SEP_ConvolutionBackpropData2D_OutputPadding,
                         ConvolutionBackpropDataSEPLayerTest_NPU3720,
                         ::testing::Combine(se_conv2DParams_OutputPadding, ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(seInputShapes)),
                                            ::testing::ValuesIn(emptyOutputShape), ::testing::Values(DEVICE_NPU)),
                         ConvolutionBackpropDataSEPLayerTest_NPU3720::getTestCaseName);

// ------ NPU4000 ------
INSTANTIATE_TEST_SUITE_P(smoke_precommit_SEP_ConvolutionBackpropData2D_ExplicitPadding,
                         ConvolutionBackpropDataSEPLayerTest_NPU4000,
                         ::testing::Combine(se_conv2DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(seInputShapes)),
                                            ::testing::ValuesIn(emptyOutputShape), ::testing::Values(DEVICE_NPU)),
                         ConvolutionBackpropDataSEPLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_SEP_ConvolutionBackpropData2D_OutputPadding,
                         ConvolutionBackpropDataSEPLayerTest_NPU4000,
                         ::testing::Combine(se_conv2DParams_OutputPadding, ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(seInputShapes)),
                                            ::testing::ValuesIn(emptyOutputShape), ::testing::Values(DEVICE_NPU)),
                         ConvolutionBackpropDataSEPLayerTest_NPU4000::getTestCaseName);

/* ============= 2D ConvolutionBackpropData with outputShape Convert to SEP Op ============= */
const std::vector<std::vector<ov::Shape>> seInputShapesWithOS = {{{1, 16, 128, 128}}};
const std::vector<ov::Shape> seSpecifiedOutputShape = {{128, 128}};

const std::vector<std::vector<size_t>> seKernelsWithOS = {{2, 2}};
const std::vector<std::vector<size_t>> seStridesWithOS = {{2, 2}};
const std::vector<std::vector<ptrdiff_t>> sePadBeginsWithOS = {{64, 64}};
const std::vector<std::vector<ptrdiff_t>> sePadEndsWithOS = {{64, 64}};
const std::vector<std::vector<size_t>> seDilationsWithOS = {{1, 1}};

const auto se_conv2DParamsWithOS_ExplicitPadding =
        ::testing::Combine(::testing::ValuesIn(seKernelsWithOS), ::testing::ValuesIn(seStridesWithOS),
                           ::testing::ValuesIn(sePadBeginsWithOS), ::testing::ValuesIn(sePadEndsWithOS),
                           ::testing::ValuesIn(seDilationsWithOS), ::testing::ValuesIn(numOutChannels),
                           ::testing::Values(ov::op::PadType::EXPLICIT), ::testing::ValuesIn(emptyOutputPadding));

// ------ NPU3720 ------
INSTANTIATE_TEST_SUITE_P(
        smoke_precommit_SEP_ConvolutionBackpropData2DWithOutputShape_ExplicitPadding,
        ConvolutionBackpropDataSEPLayerTest_NPU3720,
        ::testing::Combine(se_conv2DParamsWithOS_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
                           ::testing::ValuesIn(static_shapes_to_test_representation(seInputShapesWithOS)),
                           ::testing::ValuesIn(seSpecifiedOutputShape), ::testing::Values(DEVICE_NPU)),
        ConvolutionBackpropDataSEPLayerTest_NPU3720::getTestCaseName);

}  // namespace
