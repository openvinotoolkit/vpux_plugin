//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#include "shared_test_classes/subgraph/mobV2_SOH.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/node_builders/group_convolution.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {

namespace test {

using namespace ov::test::utils;

std::string mobilenetV2SlicedTest::getTestCaseName(const testing::TestParamInfo<mobilenetV2SlicedParameters>& obj) {
    ov::element::Type modelType;
    std::map<std::string, std::string> configuration;
    std::tie(modelType, std::ignore, configuration) = obj.param;

    std::ostringstream result;
    result << "netPRC=" << modelType << "_";
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void mobilenetV2SlicedTest::SetUp() {
    /* creates subgraph
           input
             |
          groupConv
             |
            Add1
             |
           Clamp
             |
           Conv
             |
            Add2
             |
           output
    */
    ov::element::Type modelType;
    std::map<std::string, std::string> tempConfig;
    std::tie(modelType, std::ignore, tempConfig) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    // input
    std::vector<size_t> inputShape = {1, 144, 56, 56};
    const auto params = std::make_shared<ov::op::v0::Parameter>(modelType, ov::Shape(inputShape));
    // GroupConv
    const auto groupConvWeights = ov::test::utils::generate_float_numbers(144 * 3 * 3, -0.2f, 0.2f);
    const auto groupConv = make_group_convolution(params, modelType, {3, 3}, {2, 2}, {1, 1}, {1, 1}, {1, 1},
                                                  ov::op::PadType::EXPLICIT, 144, 144, false, groupConvWeights);
    // Add1
    const std::vector<float> bias = ov::test::utils::generate_float_numbers(144, -5.f, 5.f);
    const auto biasNode = ov::op::v0::Constant::create(modelType, ov::Shape{1, 144, 1, 1}, bias);
    const auto add1 = std::make_shared<ov::op::v1::Add>(groupConv, biasNode);
    // Clamp
    const auto clamp = std::make_shared<ov::op::v0::Clamp>(add1, 0.0f, 6.0f);
    // conv
    std::vector<size_t> convInputShape = {1, 144, 28, 28};
    const auto convWeights = ov::test::utils::generate_float_numbers(32 * convInputShape[1] * 1 * 1, -0.2f, 0.2f);
    const auto conv = make_convolution(clamp, modelType, {1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                       ov::op::PadType::EXPLICIT, 32, false, convWeights);

    // Add2
    const std::vector<float> bias1 = ov::test::utils::generate_float_numbers(32, -5.f, 5.f);
    const auto biasNode1 = ov::op::v0::Constant::create(modelType, ov::Shape{1, 32, 1, 1}, bias1);
    const auto add2 = std::make_shared<ov::op::v1::Add>(conv, biasNode1);

    // result
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add2)};

    function = std::make_shared<ov::Model>(results, ov::ParameterVector{params}, "MobilenetV2SlicedTest");
}
}  // namespace test
}  // namespace ov
