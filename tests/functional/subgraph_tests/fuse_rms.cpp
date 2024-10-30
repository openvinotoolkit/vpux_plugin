// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <ov_ops/rms.hpp>
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset6.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
using namespace ov::test;
namespace ov::test::subgraph {

using RMSNormDecompositionParams = std::tuple<ov::Shape,           // input shapes
                                              std::vector<float>,  // gamma
                                              ov::element::Type>;  // input precision

class FuseRMSTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<RMSNormDecompositionParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<RMSNormDecompositionParams> obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 10, 1, 100);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }

    std::shared_ptr<ov::Model> init_subgraph(std::vector<ov::PartialShape>& input_shapes, const ov::Shape& target_shape,
                                             const ov::element::Type input_precision) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(input_precision, input_shapes[0])};

        // x^2
        auto power_const = ov::op::v0::Constant::create(input_precision, {}, {2.f});
        auto power = std::make_shared<ov::op::v1::Power>(params[0], power_const);

        // ReduceMean(x^2,axes)
        auto mean_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto mean = std::make_shared<ov::op::v1::ReduceMean>(power, mean_axes, true);

        // ReduceMean(x^2,axes)+eps
        auto eps = ov::op::v0::Constant::create(input_precision, {}, {1e-5f});
        auto add_eps = std::make_shared<ov::op::v1::Add>(mean, eps);

        // Sqrt(ReduceMean(x^2,axes)+eps)
        auto sqrt = std::make_shared<ov::op::v0::Sqrt>(add_eps);

        // 1/Sqrt(ReduceMean(x^2,axes)+eps)
        auto div_const = ov::op::v0::Constant::create(input_precision, {}, {1});
        auto div = std::make_shared<ov::op::v1::Divide>(div_const, sqrt);

        // x * 1/Sqrt(ReduceMean(x^2,axes)+eps)
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(params[0], div);

        // x * 1/Sqrt(ReduceMean(x^2,axes)+eps) * gamma
        auto dim = *target_shape.rbegin();

        auto tensor = ov::test::utils::create_and_fill_tensor(input_precision, ov::Shape{dim});
        auto gamma = std::make_shared<ov::op::v0::Constant>(tensor);
        auto mul2 = std::make_shared<ov::op::v1::Multiply>(gamma, mul1);

        auto comp = std::make_shared<ov::op::v0::Convert>(mul2, ov::element::f16);

        return std::make_shared<ov::Model>(ov::NodeVector{comp}, params, "RMSNormDecomposition");
    }
    void SetUp() override {
        ov::Shape input_shapes;
        std::vector<float> gamma;
        ov::element::Type input_precision;

        std::tie(input_shapes, gamma, input_precision) = GetParam();
        inType = outType = input_precision;
        init_input_shapes(ov::test::static_shapes_to_test_representation({input_shapes}));
        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(input_precision, inputDynamicShapes.front())};

        auto rms_const = ov::opset10::Constant::create(ov::element::f32, {gamma.size()}, gamma);
        auto rms = std::make_shared<ov::op::internal::RMS>(params[0], rms_const, 1e-5f, ov::element::f16);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(rms)};

        function = std::make_shared<ov::Model>(results, params, "fuse_rms");
    }
};

TEST_P(FuseRMSTestCommon, NPU3720_HW) {
    abs_threshold = 0.1f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(FuseRMSTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
namespace {
const std::vector<ov::element::Type> input_precisions = {ov::element::f32};

const std::vector<ov::Shape> input_shapes_basic = {{{1, 2, 6}}, {{2, 2, 6}}};
std::vector<float> gamma_basic = {0.029f, 0.014f, 0.003f, 0.013f, 0.015f, 0.009f};

const std::vector<ov::Shape> input_shapes = {{{1, 2, 16}}, {{1, 4, 16, 16}}};
std::vector<float> gamma = {0.029785f, 0.014038f, 0.003098f, 0.013123f, 0.015137f, 0.009399f, 0.008362f, 0.008179f,
                            0.018188f, 0.021973f, 0.005249f, 0.004639f, 0.004272f, 0.020264f, 0.013489f, 0.008789f};

INSTANTIATE_TEST_SUITE_P(precommit_FuseRMS, FuseRMSTestCommon,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_basic), ::testing::Values(gamma_basic),
                                            ::testing::ValuesIn(input_precisions)),
                         FuseRMSTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FuseRMS, FuseRMSTestCommon,
                         ::testing::Combine(::testing::ValuesIn(input_shapes), ::testing::Values(gamma),
                                            ::testing::ValuesIn(input_precisions)),
                         FuseRMSTestCommon::getTestCaseName);

}  // namespace
}  // namespace ov::test::subgraph
