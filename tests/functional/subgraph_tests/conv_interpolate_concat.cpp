//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

using ConvInterpolateConcatTestParams = std::tuple<std::vector<size_t>,  // input shape
                                                   int64_t               // concat axis
                                                   >;

class ConvInterpolateConcatTestCommon :
        public testing::WithParamInterface<ConvInterpolateConcatTestParams>,
        public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ConvInterpolateConcatTestParams> obj) {
        std::vector<size_t> inputShape;
        int axis;
        std::tie(inputShape, axis) = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "inputShape={" << inputShape.at(0) << ", " << inputShape.at(1) << ", " << inputShape.at(2) << ", "
               << inputShape.at(3) << "}" << sep;
        result << "axis={" << axis << "}_";
        return result.str();
    }

    std::shared_ptr<ov::Node> buildConv(const ov::Output<ov::Node>& param) {
        const ov::Shape inputShape = param.get_shape();
        const auto weightsSize = 16 * inputShape.at(1) * 3 * 3;
        std::vector<float> values(weightsSize, 1.f);
        const auto weightsShape = ov::Shape{16, inputShape.at(1), 3, 3};
        const auto weights = ov::op::v0::Constant::create(ov::element::f16, weightsShape, values);

        const ov::Strides strides = {2, 2};
        const ov::CoordinateDiff pads_begin = {1, 1};
        const ov::CoordinateDiff pads_end = {1, 1};
        const ov::Strides dilations = {1, 1};
        auto conv2d_node = std::make_shared<ov::op::v1::Convolution>(param, weights->output(0), strides, pads_begin,
                                                                     pads_end, dilations);

        return conv2d_node;
    }

    std::shared_ptr<ov::Node> buildInterpolate(const ov::Output<ov::Node>& param) {
        std::vector<float> scales = {1.f, 1.f, 2.f, 2.f};
        auto scales_const = ov::op::v0::Constant(ov::element::Type_t::f32, ov::Shape{scales.size()}, scales);
        auto scalesInput = std::make_shared<ov::op::v0::Constant>(scales_const);

        ov::op::util::InterpolateBase::InterpolateMode mode = ov::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX;
        ov::op::util::InterpolateBase::ShapeCalcMode shapeCalcMode = ov::op::v4::Interpolate::ShapeCalcMode::SCALES;
        std::vector<size_t> padBegin = {0, 0, 0, 0};
        std::vector<size_t> padEnd = {0, 0, 0, 0};
        ov::op::util::InterpolateBase::CoordinateTransformMode coordinateTransformMode =
                ov::op::v4::Interpolate::CoordinateTransformMode::ASYMMETRIC;
        ov::op::util::InterpolateBase::NearestMode nearestMode =
                ov::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
        bool antialias = false;
        double cubeCoef = -0.75;

        ov::op::util::InterpolateBase::InterpolateAttrs interpolateAttributes{
                mode, shapeCalcMode, padBegin, padEnd, coordinateTransformMode, nearestMode, antialias, cubeCoef};

        return std::make_shared<ov::op::v11::Interpolate>(param, scalesInput, interpolateAttributes);
    }

    void SetUp() override {
        std::vector<size_t> lhsInputShapeVec;
        int axis;
        std::tie(lhsInputShapeVec, axis) = this->GetParam();

        std::vector<size_t> rhsInputShapeVec = lhsInputShapeVec;
        InputShape lhsInputShape = {{}, std::vector<ov::Shape>({lhsInputShapeVec})};
        InputShape rhsInputShape = {{}, std::vector<ov::Shape>({rhsInputShapeVec})};
        init_input_shapes({lhsInputShape, rhsInputShape});

        auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes[0]);
        input0->set_friendly_name("input_0");

        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes[1]);
        input1->set_friendly_name("input_1");

        auto conv0 = buildConv(input0);
        auto conv1 = buildConv(input1);

        auto interp0 = buildInterpolate(conv0);
        auto interp1 = buildInterpolate(conv1);

        auto concat =
                std::make_shared<ov::op::v0::Concat>(ov::OutputVector({interp0->output(0), interp1->output(0)}), axis);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(concat)};
        function =
                std::make_shared<ov::Model>(results, ov::ParameterVector{input0, input1}, "ConvInterpolateConcatTest");

        rel_threshold = 0.1f;
    }
};

class ConvInterpolateConcatTest_NPU3700 : public ConvInterpolateConcatTestCommon {};

TEST_P(ConvInterpolateConcatTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

class ConvInterpolateConcatTest_NPU3720 : public ConvInterpolateConcatTestCommon {};

TEST_P(ConvInterpolateConcatTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

class ConvInterpolateConcatTest_NPU4000 : public ConvInterpolateConcatTestCommon {};

TEST_P(ConvInterpolateConcatTest_NPU4000, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace ov::test::subgraph

using namespace ov::test::subgraph;

namespace {

const std::vector<std::vector<size_t>> inputShapes = {
        {1, 16, 224, 224},
};

const std::vector<int64_t> concatAxis = {1, 2, 3};

INSTANTIATE_TEST_SUITE_P(smoke_ConvInterpolateConcatTest, ConvInterpolateConcatTest_NPU3700,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(concatAxis)),
                         ConvInterpolateConcatTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvInterpolateConcatTest, ConvInterpolateConcatTest_NPU3720,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(concatAxis)),
                         ConvInterpolateConcatTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ConvInterpolateConcatTest, ConvInterpolateConcatTest_NPU4000,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::ValuesIn(concatAxis)),
                         ConvInterpolateConcatTestCommon::getTestCaseName);

}  // namespace
