//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace {
struct GroupQuantShapes {
    const ov::Shape _lhsShape;
    const ov::Shape _weightShape;
    const ov::Shape _scaleShape;
    const std::vector<int64_t> _rhsShape;
    const bool _transposeB;
};
using GroupQuantParams = std::tuple<GroupQuantShapes, ov::element::Type, bool>;
}  // namespace

namespace ov::test::subgraph {

class MatMulWithFQTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<GroupQuantParams> {
public:
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        OPENVINO_ASSERT(inputShapes.size() == 1, "Only 1 input shape is supported");
        const auto& funcInputs = function->inputs();
        OPENVINO_ASSERT(funcInputs.size() == 1, "Only 1 input is supported");
        const auto& inputStaticShape = inputShapes[0];
        const auto totalSize = ov::shape_size(inputStaticShape);
        auto inputTensor = ov::Tensor{ov::element::f32, inputStaticShape};
        auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f32>::value_type>();
        for (size_t i = 0; i < totalSize; i++) {
            inputData[i] = std::sin(i);
        }
        inputs = {
                {funcInputs[0].get_node_shared_ptr(), inputTensor},
        };
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

    void SetUp() override {
        // Create a subgraph that will be lowered into a FakeQuantize with two axes
        // FQ (3x32x64 * 3x1x64) -> Reshape (3x32x64 to 96x64) -> MatMul (16x96 * 96x64)
        const auto& [shapes, weigthsType, hasOutputScaleShiftU16Fq] = GetParam();
        const std::vector<ov::Shape> inferenceShapes = {shapes._lhsShape};
        const ov::test::InputShape dataShape = {shapes._lhsShape, inferenceShapes};
        init_input_shapes({dataShape});
        const auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, inputDynamicShapes.at(0));
        const auto& weightShape = shapes._weightShape;
        const auto weightTotalSize = ov::shape_size(weightShape);
        // Limit the value range.
        // For I8 the weights lie within [0, 128) half-open interval.
        // For I4 and U4 the values belong to [0, 8) interval.
        // I8 bitwidth = 8, 1 << (bitwidth - 1) = 1 << 7 = 10000000b = 128
        // U4 bitwidth = 4, 1 << (bitwidth - 1) = 1 << 3 = 1000b = 8
        const auto upperBound = 1 << (weigthsType.bitwidth() - 1);
        const std::map<ov::element::Type, float> zeroPointsMap = {
                {ov::element::u4, 8},
                {ov::element::i4, 0},
                {ov::element::i8, 1},
        };
        const auto zeroPoint = zeroPointsMap.at(weigthsType);
        std::vector<int8_t> weightsData(weightTotalSize, 0);
        for (size_t i = 0; i < weightsData.size(); i++) {
            weightsData.at(i) = i % upperBound;
        }
        const auto weights = ov::opset1::Constant::create(weigthsType, weightShape, weightsData);
        const auto convert = std::make_shared<ov::opset1::Convert>(weights->output(0), ov::element::f32);

        const auto& scaleShiftShape = shapes._scaleShape;
        const auto scaleShiftTotalSize = ov::shape_size(scaleShiftShape);
        using scaleShiftValueType = ov::element_type_traits<ov::element::f32>::value_type;
        const std::vector<scaleShiftValueType> zeroPointData(scaleShiftTotalSize, zeroPoint);
        const auto zeroPoints = ov::opset1::Constant::create(ov::element::f32, scaleShiftShape, zeroPointData);
        const auto shift = std::make_shared<ov::opset1::Subtract>(convert->output(0), zeroPoints->output(0));

        std::vector<scaleShiftValueType> scaleData(scaleShiftTotalSize, 0);
        for (size_t i = 0; i < scaleData.size(); i++) {
            scaleData.at(i) = ((i % 7) + 2) / 128.f;
        }
        const auto scales = ov::opset1::Constant::create(ov::element::f32, scaleShiftShape, scaleData);
        const auto mul = std::make_shared<ov::opset1::Multiply>(shift->output(0), scales->output(0));

        const auto& matrixShape = shapes._rhsShape;
        const auto targetShape =
                ov::opset1::Constant::create(ov::element::i64, ov::Shape{matrixShape.size()}, matrixShape);
        const auto reshape = std::make_shared<ov::opset1::Reshape>(mul->output(0), targetShape->output(0), false);

        const auto matmul =
                std::make_shared<ov::opset1::MatMul>(param->output(0), reshape->output(0), false, shapes._transposeB);

        auto output = matmul->output(0);
        if (hasOutputScaleShiftU16Fq) {
            float inLow = -128.f;
            float inHigh = 127.f;
            float scale = 2.f;
            float bias = 5.f;
            float outLow = inLow * scale + bias;
            float outHigh = inHigh * scale + bias;
            const auto u16FqInLow = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1}, {inLow});
            const auto u16FqInHigh = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1}, {inHigh});
            const auto u16FqOutLow = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1}, {outLow});
            const auto u16FqOutHigh = ov::opset1::Constant::create(ov::element::f32, ov::Shape{1}, {outHigh});
            constexpr size_t u16FqLevels = 65536;
            const auto u16Fq = std::make_shared<ov::opset1::FakeQuantize>(
                    matmul->output(0), u16FqInLow->output(0), u16FqInHigh->output(0), u16FqOutLow->output(0),
                    u16FqOutHigh->output(0), u16FqLevels, ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY));
            output = u16Fq->output(0);
        }

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(output)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "MatMulWithFQ");
    }

    static std::string getTestCaseName(const testing::TestParamInfo<GroupQuantParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        const auto& [shapes, dataType, hasOutputScaleShiftU16Fq] = obj.param;
        result << "DataType=" << dataType << sep;
        result << "DataShape=" << shapes._lhsShape << sep;
        result << "WeightShape=" << shapes._weightShape << sep;
        result << "hasOutputScaleShiftU16Fq=" << hasOutputScaleShiftU16Fq;
        return result.str();
    };
};

//
// Platform test definition
//

TEST_P(MatMulWithFQTestCommon, NPU3720_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(MatMulWithFQTestCommon, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

class MatMulWithFQTestEnableU16FqScaleShiftConversion : public MatMulWithFQTestCommon {
    void configure_model() override {
        configuration[ov::intel_npu::compilation_mode_params.name()] =
                "enable-u16-fake-quantize-to-scale-shift-conversion=true";
    }
};

TEST_P(MatMulWithFQTestEnableU16FqScaleShiftConversion, NPU4000_TestKindSubgraph) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

const std::vector<GroupQuantShapes> shapes = {
        /*case1=*/{
                /*_lhsShape=*/{16, 3 * 32},
                /*_weightShape=*/{3, 32, 64},
                /*_scaleShape=*/{3, 1, 64},
                /*_rhsShape=*/{3 * 32, 64},
                /*_transposeB=*/false,
        },
        /*case2=*/
        {
                /*_lhsShape=*/{1, 16, 3 * 32},
                /*_weightShape=*/{64, 3, 32},
                /*_scaleShape=*/{64, 3, 1},
                /*_rhsShape=*/{64, 3 * 32},
                /*_transposeB=*/true,
        },
};
const std::vector<ov::element::Type> elementTypes = {ov::element::i8, ov::element::i4, ov::element::u4};

INSTANTIATE_TEST_SUITE_P(MatMulWithFQ, MatMulWithFQTestCommon,
                         ::testing::Combine(::testing::ValuesIn(shapes), ::testing::ValuesIn(elementTypes),
                                            ::testing::Values(false)),
                         MatMulWithFQTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MatMulWithFQ, MatMulWithFQTestEnableU16FqScaleShiftConversion,
                         ::testing::Combine(::testing::ValuesIn(shapes), ::testing::ValuesIn(elementTypes),
                                            ::testing::Values(true)),
                         MatMulWithFQTestEnableU16FqScaleShiftConversion::getTestCaseName);

}  // namespace ov::test::subgraph
