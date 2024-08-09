//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#include "shared_test_classes/subgraph/conv_act_base.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/node_builders/activation.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {

namespace test {

using namespace ov::test::utils;

std::string ConvActTest::getTestCaseName(const testing::TestParamInfo<convActTestParamsSet>& obj) {
    activationParams aParams;
    ov::op::PadType padType;
    std::vector<size_t> kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    std::tie(aParams, kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = obj.param;

    const std::string sep = "_";
    std::ostringstream result;
    result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
    result << "TestIdx=" << obj.index << sep;
    auto accPartName = ActivationLayerTest::getTestCaseName({aParams, 0});

    result << "K" << vec2str(kernel) << sep;
    result << "S" << vec2str(stride) << sep;
    result << "PB" << vec2str(padBegin) << sep;
    result << "PE" << vec2str(padEnd) << sep;
    result << "D=" << vec2str(dilation) << sep;
    result << "O=" << convOutChannels << sep;
    result << "AP=" << padType << sep;
    result << accPartName;
    return result.str();
}

void ConvActTest::buildFloatFunction() {
    auto modelType = ov::element::undefined;
    ov::op::PadType padType;
    std::vector<size_t> kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;

    activationParams aParams;
    std::tie(aParams, kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = GetParam();

    std::pair<std::vector<InputShape>, ov::Shape> shapes;
    std::pair<ActivationTypes, std::vector<float>> activationDecl;
    std::tie(activationDecl, modelType, shapes, std::ignore) = aParams;
    ov::Shape inputShapeFirst = shapes.first[0].second[0];
    init_input_shapes(shapes.first);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(modelType, inputDynamicShapes.front())};

    std::vector<float> filter_weights;
    auto filter_size = std::accumulate(std::begin(kernel), std::end(kernel), 1, std::multiplies<size_t>());
    filter_weights =
            ov::test::utils::generate_float_numbers(convOutChannels * inputShapeFirst[1] * filter_size, -0.5f, 0.5f);
    auto conv = std::dynamic_pointer_cast<ov::op::v1::Convolution>(
            make_convolution(params[0], modelType, kernel, stride, padBegin, padEnd, dilation, padType, convOutChannels,
                             false, filter_weights));

    ov::ResultVector results{};

    ActivationTypes activationType;
    activationType = activationDecl.first;
    auto constantsValue = activationDecl.second;
    auto activation = make_activation(conv, modelType, activationType, shapes.second, constantsValue);
    results.push_back(std::make_shared<ov::op::v0::Result>(activation));

    function = std::make_shared<ov::Model>(results, params, "convolution");
}

void ConvActTest::buildFQFunction() {
    auto modelType = ov::element::undefined;
    ov::op::PadType padType;
    std::vector<size_t> kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;

    activationParams aParams;
    std::tie(aParams, kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = GetParam();

    std::pair<std::vector<ov::test::InputShape>, ov::Shape> shapes;

    std::pair<ActivationTypes, std::vector<float>> activationDecl;
    std::tie(activationDecl, modelType, shapes, std::ignore) = aParams;
    ov::Shape inputShapeFirst = shapes.first[0].second[0];
    init_input_shapes(shapes.first);  // {1, 3, 62, 62};
    /// building conv+activation+FQs subgraph

    auto filter_size = std::accumulate(std::begin(kernel), std::end(kernel), 1, std::multiplies<size_t>());

    const std::vector<size_t> weightsShape{convOutChannels * filter_size, inputShapeFirst[1], 1, 1};

    const ov::ParameterVector params{
            std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

    /// building data FQ
    const size_t dataLevels = 256;
    const std::vector<float> dataLow = {0.f};
    const std::vector<float> dataHigh = {254.125f};
    const auto dataFq =
            make_fake_quantize(params[0], ov::element::f32, dataLevels, {}, dataLow, dataHigh, dataLow, dataHigh);

    /// building weights FQ - through convert layer
    const auto weightsU8 = make_constant(ov::element::u8, weightsShape, ov::test::utils::InputGenerateData(1, 255));
    const auto weightsFP32 = std::make_shared<ov::op::v0::Convert>(weightsU8, ov::element::f32);

    const size_t weightsLevels = 255;

    const auto weightsInLow = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, std::vector<float>{0.0f});
    const auto weightsInHigh = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, std::vector<float>{254.0f});

    std::vector<float> perChannelLow(weightsShape[0]);
    std::vector<float> perChannelHigh(weightsShape[0]);

    for (size_t i = 0; i < weightsShape[0]; ++i) {
        perChannelLow[i] = -1.0f;
        perChannelHigh[i] = 1.0f;
    }

    const auto weightsOutLow =
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{weightsShape[0], 1, 1, 1}, perChannelLow);
    const auto weightsOutHigh =
            ov::op::v0::Constant::create(ov::element::f32, ov::Shape{weightsShape[0], 1, 1, 1}, perChannelHigh);

    const auto weightsFq = std::make_shared<ov::op::v0::FakeQuantize>(weightsFP32, weightsInLow, weightsInHigh,
                                                                      weightsOutLow, weightsOutHigh, weightsLevels);

    /// building convolution
    const ov::Strides strides = {1, 1};
    const ov::CoordinateDiff pads_begin = {0, 0};
    const ov::CoordinateDiff pads_end = {0, 0};
    const ov::Strides dilations = {1, 1};
    const auto conv =
            std::make_shared<ov::op::v1::Convolution>(dataFq, weightsFq, strides, pads_begin, pads_end, dilations);

    /// building activation
    ActivationTypes activationType;
    activationType = activationDecl.first;
    auto constantsValue = activationDecl.second;
    auto activation = make_activation(conv, modelType, activationType, shapes.second, constantsValue);

    /// activation FQ
    const std::vector<float> outDataLow = {-14.0f};
    const std::vector<float> outDataHigh = {14.0f};
    const auto activationaFq = make_fake_quantize(activation, ov::element::f32, dataLevels, {}, outDataLow, outDataHigh,
                                                  outDataLow, outDataHigh);

    const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(activationaFq)};
    function = std::make_shared<ov::Model>(results, params, "QuantizedConvAcc");

    rel_threshold = 0.4f;
}

void ConvActTest::SetUp() {
    auto accParams = std::get<0>(GetParam());
    ov::element::Type modelType = std::get<1>(accParams);

    switch (modelType) {
    case ov::element::f32:
    case ov::element::f16:
        buildFloatFunction();
        return;
    case ov::element::u8:
        buildFQFunction();
        return;
    default:
        FAIL() << "unsupported network precision for test case: " << modelType;
    }
}
}  // namespace test
}  // namespace ov
