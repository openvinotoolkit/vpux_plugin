//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov::test {

//
//          [Conv]
//             |
//       [FakeQuantize]
//             |
//      [TransposedConv] --- [FakeQuantize] -- [DeclareOp]
//             |
//          [output]
//

struct TransposedConvFQTestParams {
    ov::Shape in_dims;
    ov::Shape w_dims;
    std::vector<uint64_t> conv_strides;
    std::vector<int64_t> conv_pads_begin;
    std::vector<int64_t> conv_pads_end;
    std::vector<uint64_t> conv_dilations;
    std::vector<uint64_t> transposed_conv_strides;
    std::vector<int64_t> transposed_conv_pads_begin;
    std::vector<int64_t> transposed_conv_pads_end;
    std::vector<int64_t> transposed_conv_out_pads;
    std::vector<uint64_t> transposed_conv_dilations;
};

class TransposedConvFQTest : public VpuOv2LayerTest, public testing::WithParamInterface<TransposedConvFQTestParams> {
    void SetUp() override {
        const auto test_params = GetParam();
        const ov::Shape inputShape = test_params.in_dims;
        const ov::Shape weightsShape = test_params.w_dims;

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        // Convolution
        const auto weightsConv = ov::op::v0::Constant::create(ov::element::f16, weightsShape, std::vector<float>{2.0f});
        const ov::Strides stridesConv = test_params.conv_strides;
        const ov::CoordinateDiff padsBeginConv = test_params.conv_pads_begin;
        const ov::CoordinateDiff padsEndConv = test_params.conv_pads_end;
        const ov::Strides dilatationConv = test_params.conv_dilations;
        const auto conv = std::make_shared<ov::op::v1::Convolution>(params[0], weightsConv, stridesConv, padsBeginConv,
                                                                    padsEndConv, dilatationConv);

        // ConvFQ
        const size_t dataLevels = 256;
        const std::vector<float> dataInputLow = {0.0f};
        const std::vector<float> dataInputHigh = {1.0f};
        const auto convFQ = ov::test::utils::make_fake_quantize(conv, ov::element::f16, dataLevels, {}, dataInputLow,
                                                                dataInputHigh, dataInputLow, dataInputHigh);

        // Weights for TransposedConv
        const auto weightsTransposedConv =
                ov::op::v0::Constant::create(ov::element::f16, weightsShape, std::vector<float>{1.0f});

        // WeightsFQ
        const std::vector<float> dataWeightsLow = {0.0f};
        const std::vector<float> dataWeightsHigh = {100.0f};
        const auto weightsFQ =
                ov::test::utils::make_fake_quantize(weightsTransposedConv, ov::element::f16, dataLevels, {},
                                                    dataWeightsLow, dataWeightsHigh, dataWeightsLow, dataWeightsHigh);

        // TransposedConvolution
        const ov::Strides stridesTransposedConv = test_params.transposed_conv_strides;
        const ov::CoordinateDiff padsBeginTransposedConv = test_params.transposed_conv_pads_begin;
        const ov::CoordinateDiff padsEndTransposedConv = test_params.transposed_conv_pads_end;
        const ov::CoordinateDiff outputPaddingTransposedConv = test_params.transposed_conv_out_pads;
        const auto autoPadTransposedConv = ov::op::PadType::EXPLICIT;
        const ov::Strides dilatationTransposedConv = test_params.transposed_conv_dilations;
        auto transposed_conv2d_node = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
                convFQ, weightsFQ, stridesTransposedConv, padsBeginTransposedConv, padsEndTransposedConv,
                dilatationTransposedConv, autoPadTransposedConv, outputPaddingTransposedConv);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(transposed_conv2d_node)};

        function = std::make_shared<ov::Model>(results, params, "TransposedConvFQTest");
        rel_threshold = 0.5f;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<TransposedConvFQTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

class TransposedConvFQTest_NPU3720 : public TransposedConvFQTest {};
class TransposedConvFQTest_NPU4000 : public TransposedConvFQTest {};

TEST_P(TransposedConvFQTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(TransposedConvFQTest_NPU4000, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

const auto transposedConvFQ5DTestCases =
        ::testing::ValuesIn<TransposedConvFQTestParams>({TransposedConvFQTestParams{
                                                                 ov::Shape{1, 16, 2, 4, 4},   // in dims
                                                                 ov::Shape{16, 16, 2, 2, 2},  // weights dims
                                                                 {1, 1, 1},                   // conv_strides
                                                                 {0, 0, 0},                   // conv_pads_begin
                                                                 {0, 0, 0},                   // conv_pads_end
                                                                 {1, 1, 1},                   // conv_dilation
                                                                 {1, 1, 1},                   // transposed_conv_strides
                                                                 {0, 0, 0},  // transposed_conv_pads_begin
                                                                 {0, 0, 0},  // transposed_conv_pads_end
                                                                 {0, 0, 0},  // transposed_conv_output_padding
                                                                 {1, 1, 1}   // transposed_conv_dilatation
                                                         },
                                                         TransposedConvFQTestParams{
                                                                 ov::Shape{1, 16, 2, 4, 4},   // in dims
                                                                 ov::Shape{16, 16, 2, 2, 2},  // weights dims
                                                                 {1, 1, 1},                   // conv_strides
                                                                 {0, 0, 0},                   // conv_pads_begin
                                                                 {0, 0, 0},                   // conv_pads_end
                                                                 {1, 1, 1},                   // conv_dilation
                                                                 {2, 2, 2},                   // transposed_conv_strides
                                                                 {0, 0, 0},  // transposed_conv_pads_begin
                                                                 {0, 0, 0},  // transposed_conv_pads_end
                                                                 {0, 0, 0},  // transposed_conv_output_padding
                                                                 {1, 1, 1}   // transposed_conv_dilatation
                                                         },
                                                         TransposedConvFQTestParams{
                                                                 ov::Shape{1, 16, 6, 4, 4},   // in dims
                                                                 ov::Shape{16, 16, 3, 2, 2},  // weights dims
                                                                 {1, 1, 1},                   // conv_strides
                                                                 {0, 0, 0},                   // conv_pads_begin
                                                                 {0, 0, 0},                   // conv_pads_end
                                                                 {1, 1, 1},                   // conv_dilation
                                                                 {1, 1, 1},                   // transposed_conv_strides
                                                                 {1, 1, 1},  // transposed_conv_pads_begin
                                                                 {0, 0, 0},  // transposed_conv_pads_end
                                                                 {0, 0, 0},  // transposed_conv_output_padding
                                                                 {1, 1, 1}   // transposed_conv_dilatation
                                                         },
                                                         TransposedConvFQTestParams{
                                                                 ov::Shape{1, 16, 6, 4, 4},   // in dims
                                                                 ov::Shape{16, 16, 3, 2, 2},  // weights dims
                                                                 {1, 1, 1},                   // conv_strides
                                                                 {0, 0, 0},                   // conv_pads_begin
                                                                 {0, 0, 0},                   // conv_pads_end
                                                                 {1, 1, 1},                   // conv_dilation
                                                                 {1, 1, 1},                   // transposed_conv_strides
                                                                 {0, 0, 0},  // transposed_conv_pads_begin
                                                                 {1, 1, 1},  // transposed_conv_pads_end
                                                                 {0, 0, 0},  // transposed_conv_output_padding
                                                                 {1, 1, 1}   // transposed_conv_dilatation
                                                         },
                                                         TransposedConvFQTestParams{
                                                                 ov::Shape{1, 16, 6, 4, 4},   // in dims
                                                                 ov::Shape{16, 16, 3, 3, 3},  // weights dims
                                                                 {1, 1, 1},                   // conv_strides
                                                                 {0, 0, 0},                   // conv_pads_begin
                                                                 {0, 0, 0},                   // conv_pads_end
                                                                 {1, 1, 1},                   // conv_dilation
                                                                 {2, 2, 2},                   // transposed_conv_strides
                                                                 {1, 1, 1},  // transposed_conv_pads_begin
                                                                 {1, 1, 1},  // transposed_conv_pads_end
                                                                 {1, 1, 1},  // transposed_conv_output_padding
                                                                 {1, 1, 1}   // transposed_conv_dilatation
                                                         },
                                                         TransposedConvFQTestParams{
                                                                 ov::Shape{1, 16, 7, 4, 4},   // in dims
                                                                 ov::Shape{16, 16, 3, 3, 3},  // weights dims
                                                                 {1, 1, 1},                   // conv_strides
                                                                 {0, 0, 0},                   // conv_pads_begin
                                                                 {0, 0, 0},                   // conv_pads_end
                                                                 {1, 1, 1},                   // conv_dilation
                                                                 {3, 3, 3},                   // transposed_conv_strides
                                                                 {0, 1, 1},  // transposed_conv_pads_begin
                                                                 {0, 1, 1},  // transposed_conv_pads_end
                                                                 {2, 2, 2},  // transposed_conv_output_padding
                                                                 {1, 1, 1}   // transposed_conv_dilatation
                                                         }});

INSTANTIATE_TEST_SUITE_P(smoke_TransposedConvFQ_5D, TransposedConvFQTest_NPU3720, transposedConvFQ5DTestCases,
                         TransposedConvFQTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TransposedConvFQ_5D, TransposedConvFQTest_NPU4000, transposedConvFQ5DTestCases,
                         TransposedConvFQTest_NPU4000::getTestCaseName);

}  // namespace ov::test
