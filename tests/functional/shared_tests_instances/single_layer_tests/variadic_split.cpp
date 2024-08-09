// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shared_test_classes/single_op/variadic_split.hpp"
#include "vpu_ov2_layer_test.hpp"

std::shared_ptr<ov::Node> makeVariadicSplit(const ov::Output<ov::Node>& in, const std::vector<size_t> numSplits,
                                            int32_t axis) {
    auto splitAxisOp =
            std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{axis});
    auto numSplit = std::make_shared<ov::op::v0::Constant>(ov::element::u64, ov::Shape{numSplits.size()}, numSplits);
    return std::make_shared<ov::op::v1::VariadicSplit>(in, splitAxisOp, numSplit);
}

namespace ov {

namespace test {

class VariadicSplitLayerTestCommon : public VariadicSplitLayerTest, virtual public VpuOv2LayerTest {};
class VariadicSplitLayerTestAxisInt32_NPU3720 : public VariadicSplitLayerTestCommon {
    void SetUp() override {
        int32_t axisInt32;
        std::vector<size_t> numSplits;
        ov::element::Type modelType;
        std::vector<InputShape> inputShape;
        std::tie(numSplits, axisInt32, modelType, inputShape, targetDevice) = this->GetParam();
        init_input_shapes(inputShape);
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(modelType, inputDynamicShapes.front())};
        auto variadicSplit = std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(
                makeVariadicSplit(params[0], numSplits, axisInt32));
        ov::ResultVector results;
        for (int i = 0; i < numSplits.size(); i++) {
            results.push_back(std::make_shared<ov::op::v0::Result>(variadicSplit->output(i)));
        }
        function = std::make_shared<ov::Model>(results, params, "VariadicSplit");
    }
};

TEST_P(VariadicSplitLayerTestCommon, NPU3720) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(VariadicSplitLayerTestAxisInt32_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(VariadicSplitLayerTestCommon, NPU4000) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test

}  // namespace ov

using ov::test::VariadicSplitLayerTestAxisInt32_NPU3720;
using ov::test::VariadicSplitLayerTestCommon;

namespace {
const std::vector<std::vector<ov::Shape>> inputShapes = {std::vector<ov::Shape>{{1, 144, 30, 40}}};

const std::vector<ov::element::Type> modelTypes = {ov::element::f32};

const std::vector<size_t> numSplits = {64, 48, 32};

const auto variadicSplitParams0 = testing::Combine(
        ::testing::Values(numSplits),                                                      // numSplits
        ::testing::Values(1),                                                              // axis
        ::testing::ValuesIn(modelTypes),                                                   // modelTypes
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes)),  // inputShapes
        ::testing::Values(ov::test::utils::DEVICE_NPU));
const auto variadicSplitParams1 =
        testing::Combine(::testing::Values(std::vector<size_t>{1, 1}),  // numSplits
                         ::testing::Values(-1),                         // axis
                         ::testing::ValuesIn(modelTypes),               // modelTypes
                         ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                 std::vector<std::vector<ov::Shape>>{{{1, 384, 2}}})),  // inputShapes
                         ::testing::Values(ov::test::utils::DEVICE_NPU));
const auto variadicSplitParams2 =
        testing::Combine(::testing::Values(std::vector<size_t>{1, 1}),  // numSplits
                         ::testing::Values(-1),                         // axis
                         ::testing::ValuesIn(modelTypes),               // modelTypes
                         ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                 std::vector<std::vector<ov::Shape>>{{{1, 384, 2}}})),  // inputShapes
                         ::testing::Values(ov::test::utils::DEVICE_NPU));
const auto variadicSplitParams3 =
        testing::Combine(::testing::Values(std::vector<size_t>{2, 4, 4}),  // numSplits
                         ::testing::Values(0, 1, 2, 3),                    // axis
                         ::testing::ValuesIn(modelTypes),
                         ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                 std::vector<std::vector<ov::Shape>>{{{10, 10, 10, 10}}})),  // inputShapes
                         ::testing::Values(ov::test::utils::DEVICE_NPU));
const auto variadicSplitParams4 =
        testing::Combine(::testing::Values(std::vector<size_t>{1, 1}),  // numSplits
                         ::testing::Values(-1),                         // axis
                         ::testing::ValuesIn(modelTypes),               // modelTypes
                         ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                 std::vector<std::vector<ov::Shape>>{{{1, 4, 2}}})),  // inputShapes
                         ::testing::Values(ov::test::utils::DEVICE_NPU));

INSTANTIATE_TEST_CASE_P(smoke_precommit_VariadicSplit, VariadicSplitLayerTestCommon, variadicSplitParams0,
                        VariadicSplitLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_VariadicSplitNegAxis0, VariadicSplitLayerTestCommon, variadicSplitParams1,
                        VariadicSplitLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_VariadicSplitNegAxis1, VariadicSplitLayerTestCommon, variadicSplitParams2,
                        VariadicSplitLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_VariadicSplitPosAxis, VariadicSplitLayerTestCommon, variadicSplitParams3,
                        VariadicSplitLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_VariadicSplitNegAxis, VariadicSplitLayerTestCommon, variadicSplitParams4,
                        VariadicSplitLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_VariadicSplitInt32Axis, VariadicSplitLayerTestAxisInt32_NPU3720, variadicSplitParams3,
                        VariadicSplitLayerTestAxisInt32_NPU3720::getTestCaseName);

}  // namespace
