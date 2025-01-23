// Copyright (C) 2023 - 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>
#include "common/functions.h"

/*Creates a graph :
Conv         Input
\             /
D2S          /
  \        /
 Eltiwise
    |
 Result
*/
namespace ov::test {

class ConvConcatReshape_D2S_NPU3720 : public VpuOv2LayerTest {
    void SetUp() override {
        inType = ov::element::f16;
        outType = ov::element::f16;
        const ov::Shape input_1Shape{1, 3, 512, 512};
        const ov::Shape input_2Shape{1, 32, 256, 256};
        const ov::Shape weightsShape{12, 32, 1, 1};
        init_input_shapes(ov::test::static_shapes_to_test_representation({input_1Shape, input_2Shape}));

        ov::ParameterVector params;
        for (const auto& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(inType, shape));
        }
        const auto weights = ov::op::v0::Constant::create(ov::element::f16, weightsShape, std::vector<float>{0.0f});
        const ov::Strides strides = {1, 1};
        const ov::CoordinateDiff pads_begin = {0, 0};
        const ov::CoordinateDiff pads_end = {0, 0};
        const ov::Strides dilations = {1, 1};
        const auto conv =
                std::make_shared<ov::op::v1::Convolution>(params[1], weights, strides, pads_begin, pads_end, dilations);
        const auto D2S = std::make_shared<ov::op::v0::DepthToSpace>(conv, "blocks_first", 2UL);
        const auto addOp = std::make_shared<ov::op::v1::Add>(D2S, params[0]);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(addOp)};
        function = std::make_shared<ov::Model>(results, params, "ConvConcatReshape_D2S_NPU3720");
        rel_threshold = 1.0f;
    }
};

TEST_F(ConvConcatReshape_D2S_NPU3720, HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}
}  // namespace ov::test
