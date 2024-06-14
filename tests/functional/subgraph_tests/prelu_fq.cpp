//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset7.hpp>

#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov::test {

class PreluFqSubGraphTest_NPU4000 : public VpuOv2LayerTest {
    void SetUp() override {
        const ov::Shape inputShape{1, 16, 320, 320};

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{
                std::make_shared<ov::opset1::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        const size_t dataLevels = 256;

        const auto negativeSlope = ov::opset1::Constant::create(ov::element::f16, {1}, std::vector<ov::float16>{0.1});
        auto postOp = std::make_shared<ov::opset7::PRelu>(params[0], negativeSlope);

        const std::vector<float> outDataLow = {0.0f};
        const std::vector<float> outDataHigh = {100.0f};
        const auto outDataFq = ov::test::utils::make_fake_quantize(postOp, ov::element::f16, dataLevels, {}, outDataLow,
                                                                   outDataHigh, outDataLow, outDataHigh);

        const ov::ResultVector results{std::make_shared<ov::opset3::Result>(outDataFq)};
        function = std::make_shared<ov::Model>(results, params, "PreluFqSubGraphTest");
        rel_threshold = 0.6f;
    }
};

TEST_F(PreluFqSubGraphTest_NPU4000, HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace ov::test
