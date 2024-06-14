// Copyright (C) 2019-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/reshape.hpp"

#include <vector>

#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {

namespace test {

class ReshapeLayerTestCommon : public ReshapeLayerTest, virtual public VpuOv2LayerTest {
private:
    void SetUp() override {
        std::vector<size_t> inputShape;
        ov::element::Type modelType;
        std::vector<int64_t> outFormShapes;
        bool specialZero;
        std::tie(specialZero, modelType, inputShape, outFormShapes, std::ignore) = this->GetParam();
        VpuOv2LayerTest::init_input_shapes(static_shapes_to_test_representation({inputShape}));

        auto param = std::make_shared<ov::op::v0::Parameter>(modelType, VpuOv2LayerTest::inputDynamicShapes.front());
        auto const_node = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{outFormShapes.size()},
                                                                 outFormShapes);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(param, const_node, specialZero);
        VpuOv2LayerTest::function =
                std::make_shared<ov::Model>(reshape->outputs(), ov::ParameterVector{param}, "Reshape");
    }
    void TearDown() override {
        VpuOv2LayerTest::TearDown();
    }
};
class ReshapeLayerTest_NPU3700 : public ReshapeLayerTestCommon {};
class ReshapeLayerTest_NPU3720 : public ReshapeLayerTestCommon {};
class ReshapeLayerTest_NPU4000 : public ReshapeLayerTestCommon {};

TEST_P(ReshapeLayerTest_NPU3700, HW) {
    VpuOv2LayerTest::setDefaultHardwareMode();
    VpuOv2LayerTest::run(Platform::NPU3700);
}

TEST_P(ReshapeLayerTest_NPU3720, SW) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU3720);
}

TEST_P(ReshapeLayerTest_NPU4000, SW) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU4000);
}

}  // namespace test

}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const auto paramCollapse1 =
        ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(modelTypes),
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>({{1, 1, 1, 100}, {1, 100, 1, 1}})),
                           ::testing::Values(std::vector<int64_t>({0, 100})), ::testing::Values(DEVICE_NPU));

const auto paramCollapse2 =
        ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(modelTypes),
                           ::testing::Values(std::vector<size_t>({1, 2, 10, 10})),
                           ::testing::Values(std::vector<int64_t>({1, 0, 100})), ::testing::Values(DEVICE_NPU));

const auto paramExpand1 =
        ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(modelTypes),
                           ::testing::Values(std::vector<size_t>({1, 2, 10, 10})),
                           ::testing::Values(std::vector<int64_t>({1, 0, 100})), ::testing::Values(DEVICE_NPU));

const auto paramExpand2 = ::testing::Combine(
        ::testing::Values(true), ::testing::ValuesIn(modelTypes), ::testing::Values(std::vector<size_t>({1, 100})),
        ::testing::Values(std::vector<int64_t>({0, 100, 1, 1})), ::testing::Values(DEVICE_NPU));

const auto paramExpand3 = ::testing::Combine(
        ::testing::Values(true), ::testing::ValuesIn(modelTypes), ::testing::Values(std::vector<size_t>({1, 2, 100})),
        ::testing::Values(std::vector<int64_t>({0, 0, 10, 10})), ::testing::Values(DEVICE_NPU));

const auto paramGeneric1 =
        ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(modelTypes),
                           ::testing::Values(std::vector<size_t>({1, 1, 1, 1000})),
                           ::testing::Values(std::vector<int64_t>({1, 1000, 1, 1})), ::testing::Values(DEVICE_NPU));

const auto paramGeneric2 = ::testing::Combine(
        ::testing::Values(true), ::testing::ValuesIn(modelTypes), ::testing::Values(std::vector<size_t>{1, 4, 2, 2}),
        ::testing::Values(std::vector<int64_t>{1, 2, 4, 2}), ::testing::Values(DEVICE_NPU));

// NPU3700
INSTANTIATE_TEST_SUITE_P(smoke_ReshapeCollapse1, ReshapeLayerTest_NPU3700, paramCollapse1,
                         ReshapeLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeCollapse2, ReshapeLayerTest_NPU3700, paramCollapse2,
                         ReshapeLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeExpand1, ReshapeLayerTest_NPU3700, paramExpand1,
                         ReshapeLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeExpand2, ReshapeLayerTest_NPU3700, paramExpand2,
                         ReshapeLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeExpand3, ReshapeLayerTest_NPU3700, paramExpand3,
                         ReshapeLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeGeneric1, ReshapeLayerTest_NPU3700, paramGeneric1,
                         ReshapeLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeGeneric2, ReshapeLayerTest_NPU3700, paramGeneric2,
                         ReshapeLayerTest_NPU3700::getTestCaseName);

// NPU3720
INSTANTIATE_TEST_SUITE_P(smoke_ReshapeCollapse1, ReshapeLayerTest_NPU3720, paramCollapse1,
                         ReshapeLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeCollapse2, ReshapeLayerTest_NPU3720, paramCollapse2,
                         ReshapeLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeExpand1, ReshapeLayerTest_NPU3720, paramExpand1,
                         ReshapeLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeExpand2, ReshapeLayerTest_NPU3720, paramExpand2,
                         ReshapeLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeExpand3, ReshapeLayerTest_NPU3720, paramExpand3,
                         ReshapeLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeGeneric1, ReshapeLayerTest_NPU3720, paramGeneric1,
                         ReshapeLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeGeneric2, ReshapeLayerTest_NPU3720, paramGeneric2,
                         ReshapeLayerTest_NPU3720::getTestCaseName);

// NPU4000
INSTANTIATE_TEST_SUITE_P(smoke_precommit_ReshapeCollapse1, ReshapeLayerTest_NPU4000, paramCollapse1,
                         ReshapeLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeCollapse2, ReshapeLayerTest_NPU4000, paramCollapse2,
                         ReshapeLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ReshapeExpand1, ReshapeLayerTest_NPU4000, paramExpand1,
                         ReshapeLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeExpand2, ReshapeLayerTest_NPU4000, paramExpand2,
                         ReshapeLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ReshapeExpand3, ReshapeLayerTest_NPU4000, paramExpand3,
                         ReshapeLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeGeneric1, ReshapeLayerTest_NPU4000, paramGeneric1,
                         ReshapeLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ReshapeGeneric2, ReshapeLayerTest_NPU4000, paramGeneric2,
                         ReshapeLayerTest_NPU4000::getTestCaseName);

}  // namespace
