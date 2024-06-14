//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov::test {

struct SubtractTestParams {
    ov::Shape input1_shape;
    ov::Shape input2_shape;
};

//
//         [DeclareOp]
//             |
//         [Subtract] --- [DeclareOp]
//             |
//          [output]
//

class SubtractSubGraphTest_NPU3720 : public VpuOv2LayerTest, public testing::WithParamInterface<SubtractTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SubtractTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
    void SetUp() override {
        const auto test_params = GetParam();

        const ov::Shape input1Shape = test_params.input1_shape;
        const ov::Shape input2Shape = test_params.input2_shape;

        init_input_shapes(static_shapes_to_test_representation({input1Shape, input2Shape}));

        ov::ParameterVector params;
        for (const auto& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f16, shape));
        }

        const auto subtract = std::make_shared<ov::op::v1::Subtract>(params[0], params[1]);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(subtract)};

        function = std::make_shared<ov::Model>(results, params, "SubtractTest");
        rel_threshold = 0.5f;
    }
};

TEST_P(SubtractSubGraphTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

INSTANTIATE_TEST_SUITE_P(smoke_subtract_same_shape_const_inputs, SubtractSubGraphTest_NPU3720,
                         ::testing::Values(SubtractTestParams{
                                 {1, 1, 2, 2},  // input1 shape
                                 {1, 1, 2, 2},  // input2 shape
                         }),
                         SubtractSubGraphTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_subtract_diff_shape_const_inputs, SubtractSubGraphTest_NPU3720,
                         ::testing::Values(SubtractTestParams{
                                 {1, 1, 2, 1},  // input1 shape
                                 {1, 1, 2, 4},  // input2 shape
                         }),
                         SubtractSubGraphTest_NPU3720::getTestCaseName);

//
//         [DeclareOp]
//             |
//       [FakeQuantize]
//             |
//         [Subtract] --- [FakeQuantize] --- [DeclareOp]
//             |
//       [FakeQuantize]
//             |
//          [output]
//

class SubtractFqSubGraphTest_NPU3720 : public VpuOv2LayerTest, public testing::WithParamInterface<SubtractTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SubtractTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
    void SetUp() override {
        const auto test_params = GetParam();

        const ov::Shape input1Shape = test_params.input1_shape;
        const ov::Shape input2Shape = test_params.input2_shape;

        init_input_shapes(static_shapes_to_test_representation({input1Shape, input2Shape}));

        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        const size_t dataLevels = 255;
        const std::vector<float> dataLow = {0.0f};
        const std::vector<float> dataHigh = {255.0f};

        const auto input1 = ov::test::utils::make_fake_quantize(params[0], ov::element::f16, dataLevels, {}, dataLow,
                                                                dataHigh, dataLow, dataHigh);
        const auto input2 = ov::test::utils::make_fake_quantize(params[0], ov::element::f16, dataLevels, {}, dataLow,
                                                                dataHigh, dataLow, dataHigh);

        const auto subtract = std::make_shared<ov::op::v1::Subtract>(input2, input2);
        const auto subtractFq = ov::test::utils::make_fake_quantize(subtract, ov::element::f16, dataLevels, {}, dataLow,
                                                                    dataHigh, dataLow, dataHigh);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(subtractFq)};

        function = std::make_shared<ov::Model>(results, params, "SubtractTest");
        rel_threshold = 0.5f;
    }
};

TEST_P(SubtractFqSubGraphTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

INSTANTIATE_TEST_SUITE_P(smoke_subtract_same_shape_const_fq_inputs, SubtractFqSubGraphTest_NPU3720,
                         ::testing::Values(SubtractTestParams{
                                 {1, 1, 2, 2},  // input1 shape
                                 {1, 1, 2, 2},  // input2 shape
                         }),
                         SubtractFqSubGraphTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_subtract_diff_shape_const_fq_inputs, SubtractFqSubGraphTest_NPU3720,
                         ::testing::Values(SubtractTestParams{
                                 {1, 1, 2, 4},  // input1 shape
                                 {1, 1, 2, 1},  // input2 shape
                         }),
                         SubtractFqSubGraphTest_NPU3720::getTestCaseName);

//
//         [DeclareOp]
//             |
//           [ReLU]
//             |
//         [Subtract] --- [ReLU] --- [DeclareOp]
//             |
//          [output]
//

class SubtractActInputsSubGraphTest_NPU3720 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<SubtractTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SubtractTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
    void SetUp() override {
        const auto test_params = GetParam();

        const ov::Shape input1Shape = test_params.input1_shape;
        const ov::Shape input2Shape = test_params.input2_shape;

        init_input_shapes(static_shapes_to_test_representation({input1Shape, input2Shape}));

        ov::ParameterVector params;
        for (const auto& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f16, shape));
        }

        const auto relu1 = std::make_shared<ov::op::v0::Relu>(params[0]);
        const auto relu2 = std::make_shared<ov::op::v0::Relu>(params[1]);

        const auto subtract = std::make_shared<ov::op::v1::Subtract>(relu1, relu2);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(subtract)};

        function = std::make_shared<ov::Model>(results, params, "SubtractTest");
        rel_threshold = 0.5f;
    }
};

TEST_P(SubtractActInputsSubGraphTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

INSTANTIATE_TEST_SUITE_P(smoke_subtract_same_shapes_act_inputs, SubtractActInputsSubGraphTest_NPU3720,
                         ::testing::Values(SubtractTestParams{
                                 {1, 1, 2, 2},  // input1 shape
                                 {1, 1, 2, 2},  // input2 shape
                         }),
                         SubtractActInputsSubGraphTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_subtract_diff_shapes_act_inputs, SubtractActInputsSubGraphTest_NPU3720,
                         ::testing::Values(SubtractTestParams{
                                 {1, 1, 2, 4},  // input1 shape
                                 {1, 1, 2, 1},  // input2 shape
                         }),
                         SubtractActInputsSubGraphTest_NPU3720::getTestCaseName);

//
//         [DeclareOp]
//             |
//           [ReLU]
//             |
//       [FakeQuantize]
//             |
//         [Subtract] --- [FakeQuantize] --- [ReLU] --- [DeclareOp]
//             |
//       [FakeQuantize]
//             |
//          [output]
//

class SubtractFqActInputsSubGraphTest_NPU3720 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<SubtractTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SubtractTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
    void SetUp() override {
        const auto test_params = GetParam();

        const ov::Shape input1Shape = test_params.input1_shape;
        const ov::Shape input2Shape = test_params.input2_shape;

        init_input_shapes(static_shapes_to_test_representation({input1Shape, input2Shape}));

        ov::ParameterVector params;
        for (const auto& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f16, shape));
        }

        const size_t dataLevels = 255;
        const std::vector<float> dataLow = {0.0f};
        const std::vector<float> dataHigh = {255.0f};

        const auto relu1 = std::make_shared<ov::op::v0::Relu>(params[0]);
        const auto relu1Fq = ov::test::utils::make_fake_quantize(relu1, ov::element::f16, dataLevels, {}, dataLow,
                                                                 dataHigh, dataLow, dataHigh);

        const auto relu2 = std::make_shared<ov::op::v0::Relu>(params[1]);
        const auto relu2Fq = ov::test::utils::make_fake_quantize(relu2, ov::element::f16, dataLevels, {}, dataLow,
                                                                 dataHigh, dataLow, dataHigh);

        const auto subtract = std::make_shared<ov::op::v1::Subtract>(relu1Fq->output(0), relu2Fq->output(0));
        const auto outFq = ov::test::utils::make_fake_quantize(subtract, ov::element::f16, dataLevels, {}, dataLow,
                                                               dataHigh, dataLow, dataHigh);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(outFq)};
        function = std::make_shared<ov::Model>(results, params, "SubtractTest");
        rel_threshold = 0.5f;
    }
};

TEST_P(SubtractFqActInputsSubGraphTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

INSTANTIATE_TEST_SUITE_P(smoke_subtract_same_shapes_fq_act_inputs, SubtractFqActInputsSubGraphTest_NPU3720,
                         ::testing::Values(SubtractTestParams{
                                 {1, 1, 2, 2},  // input1 shape
                                 {1, 1, 2, 2},  // input2 shape
                         }),
                         SubtractFqActInputsSubGraphTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_subtract_diff_shapes_fq_act_inputs, SubtractFqActInputsSubGraphTest_NPU3720,
                         ::testing::Values(SubtractTestParams{
                                 {1, 1, 2, 1},  // input1 shape
                                 {1, 1, 2, 4},  // input2 shape
                         }),
                         SubtractFqActInputsSubGraphTest_NPU3720::getTestCaseName);

}  // namespace ov::test
