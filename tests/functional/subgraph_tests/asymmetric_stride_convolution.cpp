//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov::test {

struct AsymmetricStrideConvSubGraphTestParams {
    ov::Shape _in_dims;
    ov::Shape _w_dims;
    std::vector<uint64_t> _strides;
    std::vector<int64_t> _pads_begin;
    std::vector<int64_t> _pads_end;
};

class AsymmetricStrideConvSubGraphTest_NPU3700 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<AsymmetricStrideConvSubGraphTestParams> {
    void SetUp() override {
        const auto test_params = GetParam();
        const ov::Shape inputShape = test_params._in_dims;
        const ov::Shape weightsShape = test_params._w_dims;

        init_input_shapes(static_shapes_to_test_representation({inputShape}));
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front());

        const size_t dataLevels = 256;
        const std::vector<float> dataLow = {0.0f};
        const std::vector<float> dataHigh = {255.0f};
        const auto dataFq = ov::test::utils::make_fake_quantize(param, ov::element::f32, dataLevels, {}, dataLow,
                                                                dataHigh, dataLow, dataHigh);

        std::vector<uint64_t> poolStridesVec = {1, 1};
        std::vector<uint64_t> poolKernelVec = {1, 1};
        const ov::Strides poolStrides = poolStridesVec;
        const ov::Shape padsBegin = {0, 0};
        const ov::Shape padsEnd = {0, 0};
        const ov::Shape poolKernel = poolKernelVec;
        const auto pool = std::make_shared<ov::op::v1::MaxPool>(dataFq, poolStrides, padsBegin, padsEnd, poolKernel);

        size_t sizeWeights = weightsShape.at(0) * weightsShape.at(1) * weightsShape.at(2) * weightsShape.at(3);
        std::vector<float> weights(sizeWeights);
        for (std::size_t i = 0; i < weights.size(); i++) {
            weights.at(i) = std::cos(i * 3.14 / 6);
        }
        auto weightsFP32 =
                std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, weightsShape, weights.data());

        const size_t weightsLevels = 255;

        const auto weightsInLow =
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, std::vector<float>{0.0f});
        const auto weightsInHigh =
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, std::vector<float>{255.0f});

        std::vector<float> perChannelLow(weightsShape[0]);
        std::vector<float> perChannelHigh(weightsShape[0]);

        for (size_t i = 0; i < weightsShape[0]; ++i) {
            perChannelLow[i] = 0.0f;
            perChannelHigh[i] = 255.0f;
        }

        const auto weightsOutLow =
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{weightsShape.at(0), 1, 1, 1}, perChannelLow);
        const auto weightsOutHigh =
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{weightsShape.at(0), 1, 1, 1}, perChannelHigh);

        const auto weightsFq = std::make_shared<ov::op::v0::FakeQuantize>(weightsFP32, weightsInLow, weightsInHigh,
                                                                          weightsOutLow, weightsOutHigh, weightsLevels);

        const ov::Strides strides = test_params._strides;
        const ov::CoordinateDiff pads_begin = test_params._pads_begin;
        const ov::CoordinateDiff pads_end = test_params._pads_end;
        const ov::Strides dilations = {1, 1};
        const auto conv =
                std::make_shared<ov::op::v1::Convolution>(pool, weightsFq, strides, pads_begin, pads_end, dilations);

        const std::vector<float> outLow = {0.0f};
        const std::vector<float> outHigh = {255.0f};
        const auto result = ov::test::utils::make_fake_quantize(conv, ov::element::f32, dataLevels, {}, outLow, outHigh,
                                                                outLow, outHigh);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(result)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "AsymmetricStrideConvSubGraphTest");

        rel_threshold = 0.1f;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<AsymmetricStrideConvSubGraphTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_P(AsymmetricStrideConvSubGraphTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

INSTANTIATE_TEST_SUITE_P(smoke_AsymmetricStrideConv, AsymmetricStrideConvSubGraphTest_NPU3700,
                         ::testing::Values(
                                 AsymmetricStrideConvSubGraphTestParams{
                                         {1, 1, 16, 16},  // in dims
                                         {2, 1, 1, 2},    // weights dims
                                         {1, 2},          // strides
                                         {0, 0},          // pads_begin
                                         {0, 0},          // pads_end
                                 },
                                 AsymmetricStrideConvSubGraphTestParams{
                                         {1, 16, 64, 64},  // in dims
                                         {16, 16, 1, 2},   // weights dims
                                         {1, 2},           // strides
                                         {0, 0},           // pads_begin
                                         {0, 0},           // pads_end
                                 }),
                         AsymmetricStrideConvSubGraphTest_NPU3700::getTestCaseName);

}  // namespace ov::test
