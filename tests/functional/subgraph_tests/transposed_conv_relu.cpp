//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>
#include "common/functions.h"

namespace ov::test {

class TransposedConvReluTest_NPU3700 : public VpuOv2LayerTest {
    void SetUp() override {
        constexpr int inChan = 3;
        constexpr int outChan = 16;
        constexpr int inWidth = 14;
        constexpr int inHeight = 8;
        constexpr int filtWidth = 8;
        constexpr int filtHeight = 8;

        const ov::Shape inputShape = {1, inChan, inHeight, inWidth};
        const ov::Shape weightsShape{inChan, outChan, filtHeight, filtWidth};

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};
        const auto weights = ov::op::v0::Constant::create(ov::element::f16, weightsShape, std::vector<float>{-1.0f});

        const ov::Strides strides = {4, 4};
        const ov::CoordinateDiff pads_begin = {2, 2};
        const ov::CoordinateDiff pads_end = {2, 2};
        const ov::CoordinateDiff output_padding = {0, 0};
        const ov::Strides dilations = {1, 1};
        const auto auto_pad = ov::op::PadType::EXPLICIT;
        auto transposed_conv2d_node = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
                params.at(0), weights, strides, pads_begin, pads_end, dilations, auto_pad, output_padding);

        auto relu_node = std::make_shared<ov::op::v0::Relu>(transposed_conv2d_node->output(0));

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(relu_node)};

        function = std::make_shared<ov::Model>(results, params, "TransposedConvReluTest");
        rel_threshold = 0.5f;
    }
};

TEST_F(TransposedConvReluTest_NPU3700, HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}
}  // namespace ov::test
