// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_ov2_layer_test.hpp"

#include <common/print_test_case_name.hpp>
#include <pretty_test_arguments.hpp>

#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/opsets/opset3.hpp>
#include <random>

using namespace ov::test;

namespace {

using BeginAndInputShape = std::pair<ov::test::InputShape, std::vector<int32_t>>;
using DynamicStridedSliceTestParams = std::tuple<BeginAndInputShape, ov::element::Type, int64_t>;

class DynamicStridedSliceLayerTest :
        public testing::WithParamInterface<DynamicStridedSliceTestParams>,
        public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DynamicStridedSliceTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        const int32_t startFrom = 0;
        const int32_t range = 3;

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                        targetInputStaticShapes[i], range, startFrom);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    std::vector<int64_t> generateConst(const ov::Shape& shape) {
        size_t totalElements = 1;
        for (size_t dim : shape) {
            totalElements *= dim;
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int64_t> dis(0, 100);

        std::vector<int64_t> randomNumbers(totalElements);
        for (size_t i = 0; i < totalElements; ++i) {
            randomNumbers[i] = dis(gen);
        }

        return randomNumbers;
    }

protected:
    void SetUp() override {
        const auto& [Inputs, type, sliceSize] = this->GetParam();
        const auto& InputShape = Inputs.first;
        const auto& ConstSize = Inputs.second;
        ov::Shape inputConstShape(ConstSize.begin(), ConstSize.end());

        init_input_shapes({InputShape});
        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes) {
            inputParams.push_back(std::make_shared<ov::op::v0::Parameter>(type, shape));
        }

        std::vector<int64_t> randomInput = generateConst(inputConstShape);
        auto inputConst = ov::op::v0::Constant::create(ov::element::i64, inputConstShape, randomInput);

        const auto paramsShape = std::get<0>(InputShape).to_shape();
        const auto inputShapeRank = inputConstShape.size();
        const std::vector<int64_t> strides(inputShapeRank, 1);
        std::vector<int64_t> ends(ConstSize.begin(), ConstSize.end());
        // Slice the tensor by width
        ends.back() = sliceSize;
        auto endParam = ov::op::v0::Constant::create(ov::element::i64, paramsShape, ends);
        auto stridesParam = ov::op::v0::Constant::create(ov::element::i64, paramsShape, strides);
        auto stridedSlice = std::make_shared<ov::op::v1::StridedSlice>(
                inputConst, inputParams[0], endParam, stridesParam, std::vector<std::int64_t>{},
                std::vector<std::int64_t>{}, std::vector<std::int64_t>{}, std::vector<std::int64_t>{});

        inputParams[0]->set_friendly_name("input");
        function = std::make_shared<ov::Model>(stridedSlice, inputParams, "DynamicStridedSlice");
    }
};

TEST_P(DynamicStridedSliceLayerTest, NPU3720_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DynamicStridedSliceLayerTest, NPU4000_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

const std::vector<ov::element::Type> inputPrecision = {ov::element::i32};
const std::vector<int64_t> sliceSize = {150};
const std::vector<BeginAndInputShape> inShapes = {
        {staticShape(1), {12}},
        {staticShape(1), {300}},
        {staticShape(3), {4, 8, 320}},
        {staticShape(4), {4, 6, 8, 320}},
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicStridedSlice, DynamicStridedSliceLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes), ::testing::ValuesIn(inputPrecision),
                                            ::testing::ValuesIn(sliceSize)),
                         DynamicStridedSliceLayerTest::getTestCaseName);

class StridedSliceWithDynamicInputLayerTest : public VpuOv2LayerTest {
public:
    void SetUp() override {
        const ov::Shape staticShape{1, 3, 16, 32};
        const std::vector<ov::Shape> inferenceShapes = {staticShape};
        const ov::PartialShape lhsDynamicShape{1, 3, 16, ov::Dimension(1, 32)};
        const ov::test::InputShape dataShape = {lhsDynamicShape, inferenceShapes};
        init_input_shapes({dataShape});
        const auto param = std::make_shared<ov::opset3::Parameter>(ov::element::f16, inputDynamicShapes.at(0));
        const std::vector<int64_t> strides{1, 1, 1, 1};
        const std::vector<int64_t> begins{0, 0, 0, 1};
        const std::vector<int64_t> ends{1, 3, 16, 30};
        const auto beginConst = ov::opset3::Constant::create(ov::element::i64, ov::Shape{4}, begins);
        const auto endConst = ov::opset3::Constant::create(ov::element::i64, ov::Shape{4}, ends);
        const auto stridesConst = ov::opset3::Constant::create(ov::element::i64, ov::Shape{4}, strides);
        auto stridedSlice = std::make_shared<ov::opset3::StridedSlice>(
                param, beginConst, endConst, stridesConst, std::vector<std::int64_t>{}, std::vector<std::int64_t>{},
                std::vector<std::int64_t>{}, std::vector<std::int64_t>{});

        const auto results = ov::ResultVector{std::make_shared<ov::opset3::Result>(stridedSlice->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "DynamicSlice");
    }
};

TEST_F(StridedSliceWithDynamicInputLayerTest, NPU3720_HW) {
    abs_threshold = 0.0f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}
}  // namespace
