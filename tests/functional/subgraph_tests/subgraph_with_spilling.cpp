// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "common_test_utils/node_builders/eltwise.hpp"

namespace ov::test {

struct ScheduleSubGraphSpillingTestParams {
    ov::Shape _in_dims;
    ov::Shape _w_dims_conv_1;
    ov::Shape _w_dims_conv_2;
    std::vector<uint64_t> _strides;
    std::vector<int64_t> _pads_begin;
    std::vector<int64_t> _pads_end;
};

// Create a simple network of DPU tasks that when scheduled will cause spilling
//                  |-> Conv1 -> Conv2 -> |
// Input -> MaxPool |                     | Eltwise -> Output
//                  |-------------------> |
class ScheduleSubGraphSpillingTest_NPU3700 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<ScheduleSubGraphSpillingTestParams> {
    void SetUp() override {
        const auto test_params = GetParam();
        const ov::Shape inputShape = test_params._in_dims;
        const ov::Shape weights1Shape = test_params._w_dims_conv_1;
        const ov::Shape weights2Shape = test_params._w_dims_conv_2;

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        std::vector<uint64_t> poolStridesVec = {1, 1};
        std::vector<uint64_t> poolKernelVec = {1, 1};
        const ov::Strides poolStrides = poolStridesVec;
        const ov::Shape padsBegin = {0, 0};
        const ov::Shape padsEnd = {0, 0};
        const ov::Shape poolKernel = poolKernelVec;
        const auto pool = std::make_shared<ov::op::v1::MaxPool>(params[0], poolStrides, padsBegin, padsEnd, poolKernel);

        const ov::Strides strides = test_params._strides;
        const ov::CoordinateDiff pads_begin = test_params._pads_begin;
        const ov::CoordinateDiff pads_end = test_params._pads_end;
        const ov::Strides dilations = {1, 1};

        std::vector<float> weights1(weights1Shape[0] * weights1Shape[1] * weights1Shape[2] * weights1Shape[3], 1);
        auto weights1FP32 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, weights1Shape, weights1.data());

        const auto conv1 =
                std::make_shared<ov::op::v1::Convolution>(pool, weights1FP32, strides, pads_begin, pads_end, dilations);

        std::vector<float> weights2(weights2Shape[0] * weights2Shape[1] * weights2Shape[2] * weights2Shape[3], 1);
        auto weights2FP32 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, weights2Shape, weights2.data());

        const auto conv2 = std::make_shared<ov::op::v1::Convolution>(conv1, weights2FP32, strides, pads_begin, pads_end,
                                                                     dilations);
        const auto eltwise = ov::test::utils::make_eltwise(pool, conv2, ov::test::utils::EltwiseTypes::ADD);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(eltwise)};
        function = std::make_shared<ov::Model>(results, params, "ScheduleSubGraphSpillingTest");
        rel_threshold = 0.1f;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<ScheduleSubGraphSpillingTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_P(ScheduleSubGraphSpillingTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

INSTANTIATE_TEST_CASE_P(smoke_ScheduleSubGraphSpilling, ScheduleSubGraphSpillingTest_NPU3700,
                        ::testing::Values(ScheduleSubGraphSpillingTestParams{
                                {1, 16, 80, 80},  // in dims
                                {32, 16, 1, 1},   // weights 1 dims
                                {16, 32, 1, 1},   // weights 2 dims
                                {1, 1},           // strides
                                {0, 0},           // pads_begin
                                {0, 0},           // pads_end
                        }),
                        ScheduleSubGraphSpillingTest_NPU3700::getTestCaseName);

}  // namespace ov::test
