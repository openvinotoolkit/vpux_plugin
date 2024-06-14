//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

using namespace ov;
using namespace element;

namespace ov::test {

class TensorIteratorSubGraphTestCommon : public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::vector<int64_t>>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

// Slice input, merged input and concat output cases
class TensorIteratorSubGraphTestCommon_NPU3720_FORWARD : public TensorIteratorSubGraphTestCommon {
    void SetUp() override {
        // Setting up test data
        inType = ov::element::f32;
        std::vector<std::vector<size_t>> exInputShapes, bodyInputShapes;
        exInputShapes = {{2, 3, 6, 10}, {4, 4, 5, 5}};
        int64_t axis = 1;
        bodyInputShapes = {{2, 1, 6, 10}, {4, 4, 5, 5}};
        const ov::Shape weightsShape{1};

        init_input_shapes(static_shapes_to_test_representation({exInputShapes[0], exInputShapes[1]}));
        ov::ParameterVector exParams, bodyInputParams;
        for (const auto& shape : exInputShapes) {
            exParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape(shape)));
        }
        for (const auto& shape : bodyInputShapes) {
            bodyInputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape(shape)));
        }
        const auto const_0 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{weightsShape}, {1.0f});
        const auto const_1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{weightsShape}, {2.0f});

        // Setting up body
        auto add_0 = std::make_shared<op::v1::Add>(bodyInputParams[0], const_0);
        auto add_1 = std::make_shared<op::v1::Add>(bodyInputParams[1], const_1);
        auto body_result_0 = std::make_shared<op::v0::Result>(add_0);
        auto body_result_1 = std::make_shared<op::v0::Result>(add_1);
        auto body_module = std::make_shared<ov::Model>(OutputVector{body_result_0, body_result_1}, bodyInputParams);

        auto tensor_iterator = std::make_shared<ov::op::v0::TensorIterator>();
        tensor_iterator->set_function(body_module);

        tensor_iterator->set_sliced_input(bodyInputParams[0], exParams[0], 0, 1, 1, -1, axis);
        tensor_iterator->get_concatenated_slices(body_result_0, 0, 1, 1, -1, axis);

        tensor_iterator->set_merged_input(bodyInputParams[1], exParams[1], body_result_1);
        tensor_iterator->get_iter_value(body_result_1);

        auto result0 = tensor_iterator->output(0);
        auto result1 = tensor_iterator->output(1);
        function = std::make_shared<ov::Model>(OutputVector{result0, result1}, exParams, "TensorIteratorTest");
    }
};

TEST_F(TensorIteratorSubGraphTestCommon_NPU3720_FORWARD, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

class TensorIteratorSubGraphTestCommon_NPU3720_REVERSE : public TensorIteratorSubGraphTestCommon {
    void SetUp() override {
        // Setting up test data
        inType = ov::element::f32;
        std::vector<std::vector<size_t>> exInputShapes, bodyInputShapes;
        exInputShapes = {{2, 3, 6, 10}, {4, 4, 5, 5}};
        int64_t axis = 1;
        bodyInputShapes = {{2, 1, 6, 10}, {4, 4, 5, 5}};
        const ov::Shape weightsShape{1};

        init_input_shapes(static_shapes_to_test_representation({exInputShapes[0], exInputShapes[1]}));
        ov::ParameterVector exParams, bodyInputParams;
        for (const auto& shape : exInputShapes) {
            exParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape(shape)));
        }
        for (const auto& shape : bodyInputShapes) {
            bodyInputParams.push_back(std::make_shared<ov::op::v0::Parameter>(inType, ov::Shape(shape)));
        }
        const auto const_0 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{weightsShape}, {1.0f});
        const auto const_1 = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{weightsShape}, {2.0f});

        // Setting up body
        auto add_0 = std::make_shared<op::v1::Add>(bodyInputParams[0], const_0);
        auto add_1 = std::make_shared<op::v1::Add>(bodyInputParams[1], const_1);
        auto body_result_0 = std::make_shared<op::v0::Result>(add_0);
        auto body_result_1 = std::make_shared<op::v0::Result>(add_1);
        auto body_module = std::make_shared<ov::Model>(OutputVector{body_result_0, body_result_1}, bodyInputParams);

        auto tensor_iterator = std::make_shared<ov::op::v0::TensorIterator>();
        tensor_iterator->set_function(body_module);

        tensor_iterator->set_sliced_input(bodyInputParams[0], exParams[0], -1, -1, 1, 0, axis);
        tensor_iterator->get_concatenated_slices(body_result_0, 0, 1, 1, -1, axis);

        tensor_iterator->set_merged_input(bodyInputParams[1], exParams[1], body_result_1);
        tensor_iterator->get_iter_value(body_result_1);

        auto result0 = tensor_iterator->output(0);
        auto result1 = tensor_iterator->output(1);
        function = std::make_shared<ov::Model>(OutputVector{result0, result1}, exParams, "TensorIteratorTest");
    }
};

TEST_F(TensorIteratorSubGraphTestCommon_NPU3720_REVERSE, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

}  // namespace ov::test
