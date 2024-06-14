// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/gather_tree.hpp"
#include <vector>
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
using ov::test::utils::InputLayerType;

namespace ov {
namespace test {

class GatherTreeLayerTestCommon : public GatherTreeLayerTest, virtual public VpuOv2LayerTest {};

class GatherTreeLayerTest_NPU3720 : public GatherTreeLayerTestCommon {};
class GatherTreeLayerTest_NPU4000 : public GatherTreeLayerTestCommon {};

void skipCompilationCallBackImpl(std::stringstream& skip, InputLayerType secInType, ov::element::Type precision) {
    if (secInType == InputLayerType::PARAMETER) {
        skip << "Unsupported secondaryInputType, OV provides scalor end_token only, but plugin only supports "
                "tensors.";
    }
    if (precision == ov::element::f32 && secInType == InputLayerType::CONSTANT) {
        skip << "FP32 precision with secondaryInputType == CONSTANT generates invalid parent_ids!";
    }
}

TEST_P(GatherTreeLayerTest_NPU3720, HW) {
    setSkipCompilationCallback([](std::stringstream& skip) {
        InputLayerType secondaryInputType = std::get<1>(GetParam());
        ov::element::Type modelType = std::get<2>(GetParam());
        skipCompilationCallBackImpl(skip, secondaryInputType, modelType);
    });
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(GatherTreeLayerTest_NPU4000, SW) {
    setSkipCompilationCallback([](std::stringstream& skip) {
        InputLayerType secondaryInputType = std::get<1>(GetParam());
        ov::element::Type modelType = std::get<2>(GetParam());
        skipCompilationCallBackImpl(skip, secondaryInputType, modelType);
    });
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

std::vector<ov::Shape> inShapes = {
        {10, 1, 100},
        {5, 1, 10},
        {3, 2, 3},
        {20, 20, 10},
};

const std::vector<InputLayerType> secondaryInputTypes = {InputLayerType::CONSTANT, InputLayerType::PARAMETER};

const std::vector<ov::element::Type> modelType = {ov::element::f32, ov::element::i32, ov::element::f16};

const auto gatherTreeArgsSubsetPrecommit =
        testing::Combine(testing::ValuesIn(inShapes),             // Input tensors shape
                         testing::ValuesIn(secondaryInputTypes),  // Secondary input type
                         testing::ValuesIn(modelType),            // Model type
                         testing::Values(DEVICE_NPU));            // Device name

INSTANTIATE_TEST_SUITE_P(precommit_gather_tree, GatherTreeLayerTest_NPU3720, gatherTreeArgsSubsetPrecommit,
                         GatherTreeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(precommit_gather_tree, GatherTreeLayerTest_NPU4000, gatherTreeArgsSubsetPrecommit,
                         GatherTreeLayerTest::getTestCaseName);

}  // namespace
