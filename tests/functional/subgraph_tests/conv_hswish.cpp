//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>
#include "common/functions.h"

namespace ov::test {

class ConvHSwishTest_NPU3700 : public VpuOv2LayerTest {
    void SetUp() override {
        constexpr int inChan = 16;
        constexpr int inWidth = 18;
        constexpr int inHeight = 18;
        constexpr int filtWidth = 5;
        constexpr int filtHeight = 5;

        inType = outType = ov::element::f16;
        const ov::Shape inputShape = {1, inChan, inHeight, inWidth};
        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::u8, inputDynamicShapes.front())};

        std::vector<uint8_t> conv_weights(inChan * filtHeight * filtWidth, 0);
        auto conv_weights_node = std::make_shared<ov::op::v0::Constant>(
                ov::element::u8, ov::Shape{inChan, 1, 1, filtHeight, filtWidth}, conv_weights.data());

        auto conv2d_node = std::make_shared<ov::op::v1::GroupConvolution>(
                params.at(0), conv_weights_node->output(0), ov::Strides(std::vector<size_t>{1, 1}),
                ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                ov::Strides(std::vector<size_t>{1, 1}));

        std::vector<uint8_t> bias_weights(inChan, 0);
        auto bias_weights_node = std::make_shared<ov::op::v0::Constant>(ov::element::u8, ov::Shape{1, inChan, 1, 1},
                                                                        bias_weights.data());
        auto bias_node = std::make_shared<ov::op::v1::Add>(conv2d_node->output(0), bias_weights_node->output(0));

        auto hswish_node = std::make_shared<ov::op::v4::HSwish>(bias_node->output(0));
        auto pool_node = std::make_shared<ov::op::v1::AvgPool>(
                hswish_node->output(0), ov::Strides{1, 1}, ov::Shape{0, 0}, ov::Shape{0, 0}, ov::Shape{14, 14}, true);

        auto mul_node = std::make_shared<ov::op::v1::Multiply>(hswish_node->output(0), pool_node->output(0));

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(mul_node->output(0))};

        function = std::make_shared<ov::Model>(results, params, "ConvHSwishTest");

        rel_threshold = 0.5f;
    }
};

TEST_F(ConvHSwishTest_NPU3700, HW_TestKindSubgraph) {
    setSkipInferenceCallback([](std::stringstream& skip) {
        if (auto var = std::getenv("IE_NPU_TESTS_RUN_INFER")) {
            skip << "Interpreter backend doesn't implement evaluate"
                    " method for OP HSwish  comparison fails";
        }
    });
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}
}  // namespace ov::test
