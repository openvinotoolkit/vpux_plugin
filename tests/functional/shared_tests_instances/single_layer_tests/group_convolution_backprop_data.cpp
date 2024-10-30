// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_op_tests/group_convolution_backprop_data.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class GroupConvBackpropLayerTestCommon : public GroupConvBackpropLayerTest, virtual public VpuOv2LayerTest {};

TEST_P(GroupConvBackpropLayerTestCommon, NPU3720_HW) {
    abs_threshold = 0.1;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(GroupConvBackpropLayerTestCommon, NPU4000_HW) {
    abs_threshold = 0.1;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const std::vector<size_t> numOutChannels = {64};
const std::vector<size_t> numGroups = {64};
const std::vector<ov::Shape> emptyOutputShape = {{}};
const std::vector<std::vector<ptrdiff_t>> emptyOutputPadding = {{}};

/* ============= 2D GroupConvolution ============= */
const std::vector<std::vector<ov::Shape>> inputShapes2D = {{{1, 64, 64, 64}}};
const std::vector<std::vector<size_t>> kernels2D = {{4, 4}};
const std::vector<std::vector<size_t>> strides2D = {{2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {{1, 1}};
const std::vector<std::vector<size_t>> dilations2D = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> outputPadding2D = {{1, 1}};

const auto groupConvBackpropData2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels2D), ::testing::ValuesIn(strides2D), ::testing::ValuesIn(padBegins2D),
        ::testing::ValuesIn(padEnds2D), ::testing::ValuesIn(dilations2D), ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups), ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding));

const auto groupConvBackpropData2DParams_OutputPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels2D), ::testing::ValuesIn(strides2D), ::testing::ValuesIn(padBegins2D),
        ::testing::ValuesIn(padEnds2D), ::testing::ValuesIn(dilations2D), ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups), ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::ValuesIn(outputPadding2D));

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData2D_ExplicitPadding, GroupConvBackpropLayerTestCommon,
                         ::testing::Combine(groupConvBackpropData2DParams_ExplicitPadding,
                                            ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes2D)),
                                            ::testing::ValuesIn(emptyOutputShape), ::testing::Values(DEVICE_NPU)),
                         GroupConvBackpropLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvBackpropData2D_OutputPadding, GroupConvBackpropLayerTestCommon,
                         ::testing::Combine(groupConvBackpropData2DParams_OutputPadding,
                                            ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(static_shapes_to_test_representation(inputShapes2D)),
                                            ::testing::ValuesIn(emptyOutputShape), ::testing::Values(DEVICE_NPU)),
                         GroupConvBackpropLayerTestCommon::getTestCaseName);

}  // namespace
