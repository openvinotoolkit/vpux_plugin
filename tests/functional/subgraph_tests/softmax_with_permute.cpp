//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov::test {

using SoftmaxWithPermuteTestParams = std::tuple<std::vector<size_t>,  // input shape
                                                int64_t,              // SoftMax axis
                                                size_t                // SoftMax channel number
                                                >;

using InputShape = std::pair<ov::PartialShape, std::vector<ov::Shape>>;

class SoftmaxWithPermuteTestCommon :
        public testing::WithParamInterface<SoftmaxWithPermuteTestParams>,
        public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<SoftmaxWithPermuteTestParams> obj) {
        std::vector<size_t> inputShape;
        int64_t axis;
        size_t softmaxChannelSize;
        std::tie(inputShape, axis, softmaxChannelSize) = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "inputShape={" << inputShape.at(0) << ", " << inputShape.at(1) << ", " << inputShape.at(2) << ", "
               << inputShape.at(3) << "}" << sep;
        result << "axis={" << axis << "}" << sep;
        result << "softmaxChannelSize={" << softmaxChannelSize << "}" << sep;
        return result.str();
    }

    void compare(const std::vector<ov::Tensor>& expectedTensors,
                 const std::vector<ov::Tensor>& actualTensors) override {
        ASSERT_EQ(actualTensors.size(), 1);
        ASSERT_EQ(expectedTensors.size(), 1);

        const auto expected = expectedTensors[0];
        const auto actual = actualTensors[0];
        ASSERT_EQ(expected.get_size(), actual.get_size());

        const float absThreshold = 0.5f;
        ov::test::utils::compare(actual, expected, absThreshold);
    }

    std::shared_ptr<ov::Node> buildConv(const ov::Output<ov::Node>& param, size_t softmaxChannelSize) {
        const ov::Shape inputShape = param.get_shape();
        const auto weightsSize = inputShape.at(1) * softmaxChannelSize * 1 * 1;
        std::vector<float> values(weightsSize, 0.0f);
        for (std::size_t i = 0; i < softmaxChannelSize; i++) {
            values.at(i * inputShape.at(1) + i % inputShape.at(1)) = 1.0f;
        }
        const auto weightsShape = ov::Shape{softmaxChannelSize, inputShape.at(1), 1, 1};
        const auto weights = ov::opset1::Constant::create(ov::element::f16, weightsShape, values);
        auto conv2d_node = std::make_shared<ov::op::v1::Convolution>(
                param, weights->output(0), ov::Strides(std::vector<size_t>{1, 1}),
                ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                ov::Strides(std::vector<size_t>{1, 1}));

        return conv2d_node;
    }

    std::shared_ptr<ov::Node> buildReshape(const ov::Output<ov::Node>& param, const std::vector<size_t>& newShape) {
        auto constNode =
                std::make_shared<ov::opset1::Constant>(ov::element::Type_t::i64, ov::Shape{newShape.size()}, newShape);
        const auto reshape = std::dynamic_pointer_cast<ov::opset1::Reshape>(
                std::make_shared<ov::opset1::Reshape>(param, constNode, false));
        return reshape;
    }

    void SetUp() override {
        std::vector<size_t> inputShapeVec;
        int64_t axis;
        size_t softmaxChannelSize;
        std::tie(inputShapeVec, axis, softmaxChannelSize) = GetParam();

        InputShape inputShape = {{}, std::vector<ov::Shape>({inputShapeVec})};
        init_input_shapes({inputShape});

        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes[0]);
        input->set_friendly_name("input");

        const auto conv = buildConv(input, softmaxChannelSize);

        const auto softMax = std::make_shared<ov::op::v1::Softmax>(conv->output(0), axis);

        auto newShape = {softmaxChannelSize, inputShapeVec[2], inputShapeVec[3]};
        const auto reshape = buildReshape(softMax->output(0), newShape);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(reshape)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input}, "SoftmaxWithPermuteTest");
    }
};

class SoftmaxWithPermuteTest_NPU3720 : public SoftmaxWithPermuteTestCommon {};

class SoftmaxWithPermuteTest_NPU4000 : public SoftmaxWithPermuteTestCommon {};

TEST_P(SoftmaxWithPermuteTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(SoftmaxWithPermuteTest_NPU4000, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

std::vector<std::vector<size_t>> inputShape = {{1, 48, 4, 76}, {1, 64, 1, 448}};
int64_t softmaxAxis = 3;
size_t softmaxChannelNum = 16;

const auto basicCases = ::testing::Combine(::testing::ValuesIn(inputShape), ::testing::Values(softmaxAxis),
                                           ::testing::Values(softmaxChannelNum));

INSTANTIATE_TEST_CASE_P(precommit, SoftmaxWithPermuteTest_NPU3720, basicCases,
                        SoftmaxWithPermuteTestCommon::getTestCaseName);

INSTANTIATE_TEST_CASE_P(precommit, SoftmaxWithPermuteTest_NPU4000, basicCases,
                        SoftmaxWithPermuteTestCommon::getTestCaseName);

}  // namespace ov::test
