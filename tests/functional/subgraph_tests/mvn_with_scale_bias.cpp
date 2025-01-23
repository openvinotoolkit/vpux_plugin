// Copyright (C) 2023 - 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

namespace ov::test {

class MVNWithScaleBiasTest_NPU3720 : public VpuOv2LayerTest, public testing::WithParamInterface<std::vector<int64_t>> {
    void SetUp() override {
        const ov::Shape inputShape = {1, 1, 1, 320};
        ov::Layout order = "NHWC";
        inType = outType = ov::element::f16;

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        const ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        auto acrossChanels = false;
        auto normalizeVariance = true;
        float epsilonF = 0.0001;
        auto mvn = std::make_shared<ov::op::v0::MVN>(params[0], acrossChanels, normalizeVariance, epsilonF);
        // OpenVINO MVN implementation implicitly adds 0th dimension to reduction axes set which is not valid behavior
        ov::AxisSet axes;
        const size_t startAxis = acrossChanels ? 1 : 2;
        const size_t numOfDims = params[0]->get_partial_shape().size();
        for (size_t i = startAxis; i < numOfDims; i++)
            axes.insert(i);
        mvn->set_reduction_axes(axes);

        const size_t scaleShiftSize = inputShape[3];
        std::vector<double> multiplyData(scaleShiftSize, 0.078740157480314959);
        const auto multiplyConst = std::make_shared<ov::op::v0::Constant>(
                ov::element::f32, ov::Shape{1, 1, 1, scaleShiftSize}, multiplyData);
        const auto multiply = std::make_shared<ov::op::v1::Multiply>(mvn, multiplyConst);

        std::vector<float> biases(scaleShiftSize, 1.0);
        for (std::size_t i = 0; i < biases.size(); i++) {
            biases.at(i) = i * 0.25f;
        }
        auto bias_weights_node = std::make_shared<ov::op::v0::Constant>(
                ov::element::f32, ov::Shape{1, 1, 1, scaleShiftSize}, biases.data());
        auto bias_node = std::make_shared<ov::op::v1::Add>(multiply, bias_weights_node->output(0));

        const auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(bias_node);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(sigmoid)};

        function = std::make_shared<ov::Model>(results, params, "MVNWithScaleBias");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(order);
        preProc.input().model().set_layout(order);
        preProc.output().tensor().set_layout(order);
        preProc.output().model().set_layout(order);
        function = preProc.build();
        rel_threshold = 0.95f;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::vector<int64_t>>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_F(MVNWithScaleBiasTest_NPU3720, HW_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}
}  // namespace ov::test
