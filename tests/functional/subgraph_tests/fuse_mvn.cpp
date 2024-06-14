// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset4.hpp"
#include "openvino/opsets/opset6.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

using FuseMVNTestParams = std::tuple<std::vector<size_t>,  // input shape
                                     std::vector<size_t>,  // target_shape
                                     std::vector<size_t>,  // ReduceMean axis
                                     bool                  // eps inside or outside
                                     >;

class FuseMVNTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<FuseMVNTestParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FuseMVNTestParams> obj) {
        ov::Shape inputShape;
        ov::Shape targetShape;
        std::vector<size_t> axis;
        bool isEpsInside;
        std::tie(inputShape, targetShape, axis, isEpsInside) = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "inputShapeSize={" << inputShape.size() << "}" << sep;
        result << "targetShapeSize={" << targetShape.size() << "}" << sep;
        result << "axisSize={" << axis.size() << "}" << sep;
        result << "isEpsInside={" << isEpsInside << "}" << sep;
        return result.str();
    }

    void SetUp() override {
        ov::Shape input_shape;
        ov::Shape target_shape;
        std::vector<size_t> axis;
        bool isEpsInside;

        std::tie(input_shape, target_shape, axis, isEpsInside) = GetParam();

        init_input_shapes(ov::test::static_shapes_to_test_representation({input_shape}));
        ov::ParameterVector params{std::make_shared<ov::opset6::Parameter>(ov::element::f32, ov::Shape(input_shape))};

        auto reshape1_const = ov::opset6::Constant::create(ov::element::i32, {target_shape.size()}, target_shape);
        auto reshape1 = std::make_shared<ov::opset6::Reshape>(params[0], reshape1_const, false);

        auto mean1_axes = ov::opset6::Constant::create(ov::element::i32, {axis.size()}, axis);
        auto mean1 = std::make_shared<ov::opset6::ReduceMean>(reshape1, mean1_axes, true);

        auto sub1 = std::make_shared<ov::opset6::Subtract>(reshape1, mean1);

        auto x_square = std::make_shared<ov::opset6::Multiply>(reshape1, reshape1);
        auto x_square_mean_axes = ov::opset6::Constant::create(ov::element::i32, {axis.size()}, axis);
        auto x_square_mean = std::make_shared<ov::opset6::ReduceMean>(x_square, mean1_axes, true);
        auto mean1_square = std::make_shared<ov::opset6::Multiply>(mean1, mean1);

        auto sub2 = std::make_shared<ov::opset6::Subtract>(x_square_mean, mean1_square);
        auto eps = ov::opset6::Constant::create(ov::element::f32, {1}, {0.000001});

        if (isEpsInside) {
            auto eps_inside_add = std::make_shared<ov::opset6::Add>(sub2, eps);
            auto eps_inside_sqrt = std::make_shared<ov::opset6::Sqrt>(eps_inside_add);
            auto divide = std::make_shared<ov::opset6::Divide>(sub1, eps_inside_sqrt);
            auto results = ov::ResultVector{std::make_shared<ov::opset6::Result>(divide->output(0))};
            function = std::make_shared<ov::Model>(results, params, "FuseMVNInsideEPS");
        } else {
            auto eps_outside_sqrt = std::make_shared<ov::opset6::Sqrt>(sub2);
            auto eps_outside_add = std::make_shared<ov::opset6::Add>(eps_outside_sqrt, eps);
            auto divide = std::make_shared<ov::opset6::Divide>(sub1, eps_outside_add);
            auto results = ov::ResultVector{std::make_shared<ov::opset6::Result>(divide->output(0))};
            function = std::make_shared<ov::Model>(results, params, "FuseMVNOutsideEPS");
        }
    }
};

class FuseMVNTest_NPU3720 : public FuseMVNTestCommon {};
class FuseMVNTest_NPU4000 : public FuseMVNTestCommon {};

TEST_P(FuseMVNTest_NPU3720, HW) {
    rel_threshold = 0.1f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(FuseMVNTest_NPU4000, HW) {
    rel_threshold = 0.1f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace ov::test::subgraph

using namespace ov::test::subgraph;

namespace {

ov::Shape inputShape = {1, 1500, 512};
ov::Shape targetShape = {1500, 512};
std::vector<size_t> axis = {1};
std::vector<bool> isEpsInside = {true, false};

const auto epsCase = ::testing::Combine(::testing::Values(inputShape), ::testing::Values(targetShape),
                                        ::testing::Values(axis), ::testing::ValuesIn(isEpsInside));

INSTANTIATE_TEST_CASE_P(precommit_FuseMVN, FuseMVNTest_NPU3720, epsCase, FuseMVNTestCommon::getTestCaseName);

INSTANTIATE_TEST_CASE_P(precommit_FuseMVN, FuseMVNTest_NPU4000, epsCase, FuseMVNTestCommon::getTestCaseName);

}  // namespace
