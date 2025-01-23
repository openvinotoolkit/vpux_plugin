//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include <vpu_ov2_layer_test.hpp>

using namespace ov;
using namespace element;

namespace ov::test {

class MaxPoolWithClampSubGraphTestCommon : public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::vector<int64_t>>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

// This is a test case that max_pool followed by a clamp and this
// clamp has 0 as the lower bound, given some negative inputs.
// The expected data flow is:
// input -> maxpool -> [... negative values ...] -> clamp(0, 6) -> [... 0, ...]
class MaxPoolWithClamp : public MaxPoolWithClampSubGraphTestCommon {
public:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        ov::Tensor tensorData = ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(),
                                                                        targetInputStaticShapes[0], 10, -10, 1, 1);
        inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }

    void SetUp() override {
        // Setting up test data
        inType = ov::element::f32;
        std::vector<std::vector<size_t>> inputShapes;
        inputShapes = {{1, 1, 2, 4}};
        const ov::Shape weightsShape{4};

        init_input_shapes(static_shapes_to_test_representation({inputShapes[0]}));
        ov::ParameterVector exParams;
        for (const auto& shape : inputShapes) {
            exParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape(shape)));
        }

        // Setting up body
        auto maxPool = std::make_shared<ov::op::v1::MaxPool>(exParams[0], ov::Strides{2, 2}, ov::Shape{0, 0},
                                                             ov::Shape{0, 0}, ov::Shape{2, 2},
                                                             ov::op::RoundingType::FLOOR, ov::op::PadType::VALID);
        const auto clamp = std::make_shared<ov::op::v0::Clamp>(maxPool, float(0), float(6));
        auto result = std::make_shared<op::v0::Result>(clamp);
        function = std::make_shared<ov::Model>(OutputVector{result}, exParams, "MaxPoolWithClampSubgraph");
    }
};

class MaxPoolWithClamp_NPU3720 : public MaxPoolWithClamp {};
TEST_F(MaxPoolWithClamp_NPU3720, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}
class MaxPoolWithClamp_NPU4000 : public MaxPoolWithClamp {};
TEST_F(MaxPoolWithClamp_NPU4000, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}
}  // namespace ov::test
