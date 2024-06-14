// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov::test {

// This test aims for:
//   - Check NCE Eltwise Add with Relu postOp and the output FakeQuantize has negative part
//
//      [input 1]
//          |
//     (FakeQuantize)
//          |
//        (Add) --- (FakeQuantize) -- [input 2]
//          |
//       (Relu)
//          |
//     (FakeQuantize)
//          |
//       [output]
//

using QuantizedEltwiseReluTestParams = std::tuple<std::vector<float>,  // input0FqRanges
                                                  std::vector<float>,  // input1FqRanges
                                                  std::vector<float>   // outputFqRanges
                                                  >;

class QuantizedEltwiseReluGraphTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<QuantizedEltwiseReluTestParams> {
    void SetUp() override {
        std::vector<float> input0FqRanges, input1FqRanges, outputFqRanges;
        std::tie(input0FqRanges, input1FqRanges, outputFqRanges) = GetParam();

        std::vector<size_t> inputShape = {1, 16, 32, 32};
        init_input_shapes(static_shapes_to_test_representation({inputShape, inputShape}));
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(inputShape)),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(inputShape))};

        const auto makeFQdata = [](const ov::Output<ov::Node>& input, const std::vector<float>& dataRanges) {
            const size_t dataLevels = 256;
            const std::vector<float> dataInLow = {dataRanges.at(0)};
            const std::vector<float> dataInHigh = {dataRanges.at(1)};
            const std::vector<float> dataOutLow = {dataRanges.at(2)};
            const std::vector<float> dataOutHigh = {dataRanges.at(3)};
            return ov::test::utils::make_fake_quantize(input, ov::element::f32, dataLevels, {}, dataInLow, dataInHigh,
                                                       dataOutLow, dataOutHigh);
        };

        const auto input0Fq = makeFQdata(params[0], input0FqRanges);
        const auto input1Fq = makeFQdata(params[1], input1FqRanges);

        const auto addOp = std::make_shared<ov::op::v1::Add>(input0Fq->output(0), input1Fq->output(0));
        const auto reluOp = std::make_shared<ov::op::v0::Relu>(addOp);

        const auto outputFq = makeFQdata(reluOp, outputFqRanges);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(outputFq)};
        function = std::make_shared<ov::Model>(results, params, "QuantizedEltwiseRelu");
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<QuantizedEltwiseReluTestParams> obj) {
        std::vector<float> fqRanges0, fqRanges1, fqRanges2;
        std::tie(fqRanges0, fqRanges1, fqRanges2) = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "Input0FQ={" << fqRanges0.at(0) << ", " << fqRanges0.at(1) << ", " << fqRanges0.at(2) << ", "
               << fqRanges0.at(3) << "}" << sep << "Input1FQ={" << fqRanges1.at(0) << ", " << fqRanges1.at(1) << ", "
               << fqRanges1.at(2) << ", " << fqRanges1.at(3) << "}" << sep << "OutputFQ={" << fqRanges2.at(0) << ", "
               << fqRanges2.at(1) << ", " << fqRanges2.at(2) << ", " << fqRanges2.at(3) << "}" << sep;
        return result.str();
    }
};

class QuantizedEltwiseReluGraphTest_NPU3720 : public QuantizedEltwiseReluGraphTestCommon {};

class QuantizedEltwiseReluGraphTest_NPU3700 : public QuantizedEltwiseReluGraphTestCommon {};

TEST_P(QuantizedEltwiseReluGraphTest_NPU3720, HW) {
    abs_threshold = 0.2;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(QuantizedEltwiseReluGraphTest_NPU3700, HW) {
    abs_threshold = 0.1;
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

std::vector<std::vector<float>> input0FqRanges = {{0.0f, 7.0f, 0.0f, 7.0f}};
std::vector<std::vector<float>> input1FqRanges = {{0.0f, 4.0f, 0.0f, 4.0f}};
std::vector<std::vector<float>> outputFqRanges = {{-9.0f, 9.0f, -9.0f, 9.0f}, {0.0f, 9.0f, 0.0f, 9.0f}};
const auto basicCases = ::testing::Combine(::testing::ValuesIn(input0FqRanges), ::testing::ValuesIn(input1FqRanges),
                                           ::testing::ValuesIn(outputFqRanges));

INSTANTIATE_TEST_SUITE_P(precommit_QuantizedEltwiseRelu, QuantizedEltwiseReluGraphTest_NPU3720, basicCases,
                         QuantizedEltwiseReluGraphTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(precommit_QuantizedEltwiseRelu, QuantizedEltwiseReluGraphTest_NPU3700, basicCases,
                         QuantizedEltwiseReluGraphTestCommon::getTestCaseName);

}  // namespace ov::test
