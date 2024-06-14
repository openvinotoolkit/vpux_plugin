// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "common_test_utils/node_builders/fake_quantize.hpp"

using namespace ov::test;
namespace {

// MLIR detects pattern quant.dcast -> op -> quant.qcast and converts it into single quantized Op
//
//      [input 1]
//          |
//     (dequantize)
//          |
//      (Multiply) --- (dequantize) -- [input 2]
//          |
//       [output]
//          |
//      (quantize)
//

using QuantizedMulTestParams = std::tuple<ov::element::Type,   // inPrc
                                          ov::element::Type,   // outPrc
                                          std::vector<float>,  // fqRanges0
                                          std::vector<float>   // fqRanges1
                                          >;

class QuantizedMulSubGraphTest_NPU3700 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<QuantizedMulTestParams> {
    void SetUp() override {
        std::vector<float> data0FQRanges, data1FQRanges;
        std::tie(inType, outType, data0FQRanges, data1FQRanges) = GetParam();
        rel_threshold = 0.1f;

        const ov::Shape shape{1, 16, 32, 32};
        init_input_shapes(static_shapes_to_test_representation({shape, shape}));

        ov::ParameterVector params;
        for (const auto& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape));
        }

        const auto makeFQdata = [](const ov::Output<ov::Node>& input, const std::vector<float>& dataRanges) {
            const size_t dataLevels = 256;
            const std::vector<float> dataInLow = {dataRanges.at(0)};
            const std::vector<float> dataInHigh = {dataRanges.at(1)};
            const std::vector<float> dataOutLow = {dataRanges.at(2)};
            const std::vector<float> dataOutHigh = {dataRanges.at(3)};
            return ov::test::utils::make_fake_quantize(input, ov::element::f32, dataLevels, {}, dataInLow, dataInHigh,
                                                       dataOutLow, dataOutHigh);
        };

        const auto data0Fq = makeFQdata(params[0], data0FQRanges);
        const auto data1Fq = makeFQdata(params[1], data1FQRanges);

        const auto mul = std::make_shared<ov::op::v1::Multiply>(data0Fq->output(0), data1Fq->output(0));

        const std::vector<float> outputRange = {0.0f, 255.0f, 0.0f, 255.0f};
        const auto outFq = makeFQdata(mul, outputRange);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(outFq)};
        function = std::make_shared<ov::Model>(results, params, "QuantizedMul");
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<QuantizedMulTestParams>& obj) {
        std::vector<float> fqRanges0, fqRanges1;
        std::tie(std::ignore, std::ignore, fqRanges0, fqRanges1) = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << "FQ0={" << fqRanges0.at(0) << ", " << fqRanges0.at(1) << ", " << fqRanges0.at(2) << ", "
               << fqRanges0.at(3) << "}" << sep << "FQ1={" << fqRanges1.at(0) << ", " << fqRanges1.at(1) << ", "
               << fqRanges1.at(2) << ", " << fqRanges1.at(3) << "}" << sep;
        return result.str();
    }
};

TEST_P(QuantizedMulSubGraphTest_NPU3700, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3700);
}

TEST_P(QuantizedMulSubGraphTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

std::vector<std::vector<float>> fqRanges = {
        {0.0f, 255.0f, 0.0f, 255.0f},
        {0.0f, 255.0f, -1.0f, 1.0f},
        {0.0f, 244.0f, -1.0f, 1.0f},
};

const std::vector<ov::element::Type> netPrecisions = {ov::element::u8, ov::element::f16};

const auto basicCases = ::testing::Combine(::testing::ValuesIn(netPrecisions), ::testing::Values(ov::element::f32),
                                           ::testing::ValuesIn(fqRanges), ::testing::ValuesIn(fqRanges));

INSTANTIATE_TEST_SUITE_P(smoke_QuantizedMul, QuantizedMulSubGraphTest_NPU3700, basicCases,
                         QuantizedMulSubGraphTest_NPU3700::getTestCaseName);

}  // namespace
