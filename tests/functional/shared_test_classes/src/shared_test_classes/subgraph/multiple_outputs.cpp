//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/multiple_outputs.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/node_builders/convolution.hpp"

using namespace ov::test::utils;
namespace ov::test {

std::string MultioutputTest::getTestCaseName(testing::TestParamInfo<multiOutputTestParams> obj) {
    ov::element::Type modelType;
    std::map<std::string, std::string> configuration;
    size_t outputChannels;
    convParams convolutionParams;
    std::vector<size_t> inputShape;
    std::vector<size_t> kernelShape;
    size_t stride;
    std::tie(modelType, std::ignore, configuration, convolutionParams, outputChannels) = obj.param;
    std::tie(inputShape, kernelShape, stride) = convolutionParams;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "KS=" << ov::test::utils::vec2str(kernelShape) << "_";
    result << "S=" << stride << "_";
    result << "OC=" << outputChannels << "_";
    result << "modelType=" << modelType << "_";
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void MultioutputTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& funcInputs = function->inputs();
    const auto& inputStaticShape = targetInputStaticShapes[0];
    const auto totalSize =
            std::accumulate(inputStaticShape.begin(), inputStaticShape.end(), 1, std::multiplies<size_t>());
    auto inputTensor = ov::Tensor{ov::element::f16, inputStaticShape};
    auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f16>::value_type>();

    for (size_t i = 0; i < totalSize; i++) {
        float value = i % 16;
        auto f16 = static_cast<ov::fundamental_type_for<ov::element::f16>>(value);
        inputData[i] = f16.to_bits();
    }
    inputs = {
            {funcInputs[0].get_node_shared_ptr(), inputTensor},
    };
}

void MultioutputTest::SetUp() {
    ov::element::Type modelType;
    std::map<std::string, std::string> tempConfig;
    convParams convolutionParams;
    size_t outputChannels;
    std::tie(modelType, std::ignore, tempConfig, convolutionParams, outputChannels) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    std::vector<size_t> inputShape;
    std::vector<size_t> kernelShape;
    size_t stride;
    std::tie(inputShape, kernelShape, stride) = convolutionParams;

    // input
    auto params = std::make_shared<ov::op::v0::Parameter>(modelType, ov::Shape(inputShape));
    // conv 1
    auto conv1Weights =
            generate_float_numbers(outputChannels * inputShape[1] * kernelShape[0] * kernelShape[1], -0.2f, 0.2f);
    auto conv1 = make_convolution(params, modelType, {kernelShape[0], kernelShape[1]}, {stride, stride}, {1, 1}, {1, 1},
                                  {1, 1}, ov::op::PadType::VALID, outputChannels, false, conv1Weights);
    // conv 2
    std::vector<size_t> conv2InputShape = {1, outputChannels, inputShape[2], inputShape[3]};
    auto conv2Weights =
            generate_float_numbers(outputChannels * conv2InputShape[1] * kernelShape[0] * kernelShape[1], -0.2f, 0.2f);
    auto conv2 = make_convolution(conv1, modelType, {kernelShape[0], kernelShape[1]}, {stride, stride}, {0, 0}, {0, 0},
                                  {1, 1}, ov::op::PadType::VALID, outputChannels, true, conv2Weights);
    // max pool
    auto pool =
            std::make_shared<ov::op::v1::MaxPool>(conv2, ov::Strides{1, 1}, ov::Shape{0, 0}, ov::Shape{0, 0},
                                                  ov::Shape{2, 2}, ov::op::RoundingType::FLOOR, ov::op::PadType::VALID);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(conv1), std::make_shared<ov::op::v0::Result>(pool)};
    function = std::make_shared<ov::Model>(results, ov::ParameterVector{params}, "MultioutputTest");
}

}  // namespace ov::test
