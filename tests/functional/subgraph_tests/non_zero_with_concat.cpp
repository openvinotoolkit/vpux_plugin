// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <openvino/op/gather_nd.hpp>
#include <openvino/op/not_equal.hpp>
#include <shared_test_classes/base/ov_subgraph.hpp>
#include <vpu_ov2_layer_test.hpp>

#include <common/print_test_case_name.hpp>
#include <pretty_test_arguments.hpp>

#include <common_test_utils/ov_tensor_utils.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset3.hpp>

using namespace ov::test;
namespace {

using NonZeroWithConcatTestParams = std::tuple<ov::test::InputShape, ov::element::Type, size_t>;

class NonZeroWithConcatTest : public testing::WithParamInterface<NonZeroWithConcatTestParams>, public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NonZeroWithConcatTestParams>& obj) {
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
        const int32_t range = 2;

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                                        targetInputStaticShapes[i], range, startFrom);
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

    void compare(const std::vector<ov::Tensor>& expectedTensors,
                 const std::vector<ov::Tensor>& actualTensors) override {
        ASSERT_EQ(actualTensors.size(), 1);
        ASSERT_EQ(expectedTensors.size(), 1);

        const auto expected = expectedTensors[0];
        const auto actual = actualTensors[0];
        ASSERT_EQ(expected.get_size(), actual.get_size());

        const float absThreshold = 0.01f;
        const float relThreshold = 0.01f;
        ov::test::utils::compare(expected, actual, absThreshold, relThreshold);
    }

protected:
    void SetUp() override {
        const auto& [inputShape, type, numInputs] = this->GetParam();

        init_input_shapes({inputShape});
        ov::ParameterVector inputParams;
        for (auto&& shape : inputDynamicShapes) {
            inputParams.push_back(std::make_shared<ov::opset1::Parameter>(type, shape));
        }

        auto nonZero = std::make_shared<ov::opset3::NonZero>(inputParams[0], ov::element::i64);
        inputParams[0]->set_friendly_name("input");

        auto convertF16 = std::make_shared<ov::opset1::Convert>(nonZero, ov::element::f16);

        ov::OutputVector params;
        for (size_t i = 0; i < numInputs; i++) {
            params.push_back(convertF16);
        }

        const auto concat = std::make_shared<ov::opset1::Concat>(params, 0);
        const auto results = ov::ResultVector{std::make_shared<ov::opset3::Result>(concat->output(0))};
        function = std::make_shared<ov::Model>(results, inputParams, "NonZeroWithConcat");
    }
};

TEST_P(NonZeroWithConcatTest, NPU3720_HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(NonZeroWithConcatTest, NPU3720_SW_TestKindSubgraph) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(NonZeroWithConcatTest, NPU4000_HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

const std::vector<ov::element::Type> inputPrecision = {ov::element::i32};

const std::vector<ov::test::InputShape> inShapes = {staticShape(4, 8)};

const std::vector<size_t> inputsNum = {
        2,
        3,
        4,
};

INSTANTIATE_TEST_SUITE_P(smoke_NonZeroWithConcat, NonZeroWithConcatTest,
                         ::testing::Combine(::testing::ValuesIn(inShapes), ::testing::ValuesIn(inputPrecision),
                                            ::testing::ValuesIn(inputsNum)),
                         NonZeroWithConcatTest::getTestCaseName);
}  // namespace
