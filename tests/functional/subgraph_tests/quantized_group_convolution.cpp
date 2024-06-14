// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/quantized_group_convolution.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "common_test_utils/node_builders/group_convolution.hpp"
#include "npu_private_properties.hpp"

#include <vector>

#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
namespace ov {
namespace test {

// MLIR detects pattern quant.dcast -> op -> quant.qcast and converts it into single quantized Op
//
//       [input]
//          |
//     (dequantize)
//          |
//        (conv) --- (dequantize) -- [filter]
//          |
//       [output]
//          |
//      (quantize)
//

class QuantGroupConvSubGraphTest_NPU3700 : public QuantGroupConvLayerTest, public VpuOv2LayerTest {
    void SetUp() override {
        VpuOv2LayerTest::rel_threshold = 0.5f;

        quantGroupConvSpecificParams groupConvParams;
        ov::Shape inputShape;
        auto modelType = ov::element::undefined;
        std::tie(groupConvParams, modelType, inputShape, std::ignore) = this->GetParam();
        ov::op::PadType padType = ov::op::PadType::AUTO;
        ov::Shape kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels, numGroups;
        size_t quantLevels;
        QuantizationGranularity quantGranularity;
        bool quantizeWeights;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, quantLevels, quantGranularity,
                 quantizeWeights) = groupConvParams;

        VpuOv2LayerTest::init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(modelType, VpuOv2LayerTest::inputDynamicShapes.front())};

        std::vector<size_t> dataFqConstShapes(inputShape.size(), 1);
        if (quantGranularity == QuantizationGranularity::Perchannel)
            dataFqConstShapes[1] = inputShape[1];
        auto dataFq = ov::test::utils::make_fake_quantize(params[0], modelType, quantLevels, dataFqConstShapes, {0},
                                                          {255}, {0}, {255});

        std::vector<size_t> weightsShapes = {convOutChannels, inputShape[1]};
        if (weightsShapes[0] % numGroups || weightsShapes[1] % numGroups)
            throw std::runtime_error("incorrect shape for QuantGroupConvolution");
        weightsShapes[0] /= numGroups;
        weightsShapes[1] /= numGroups;
        weightsShapes.insert(weightsShapes.begin(), numGroups);
        weightsShapes.insert(weightsShapes.end(), kernel.begin(), kernel.end());

        std::vector<float> weightsData;
        std::shared_ptr<ov::Node> weights;
        if (quantizeWeights) {
            std::vector<size_t> fqWeightsShapes{convOutChannels, inputShape[1] / numGroups};
            fqWeightsShapes.insert(fqWeightsShapes.end(), kernel.begin(), kernel.end());

            std::vector<size_t> weightsFqConstShapes(inputShape.size(), 1);
            if (quantGranularity == QuantizationGranularity::Perchannel)
                weightsFqConstShapes[0] = fqWeightsShapes[0];

            auto weightsNode = ov::test::utils::deprecated::make_constant(modelType, fqWeightsShapes, weightsData,
                                                                          weightsData.empty());
            auto fqNode = ov::test::utils::make_fake_quantize(weightsNode, modelType, quantLevels, weightsFqConstShapes,
                                                              {0}, {255}, {0}, {255});

            auto constNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{weightsShapes.size()},
                                                                    weightsShapes);
            weights = std::dynamic_pointer_cast<ov::op::v1::Reshape>(
                    std::make_shared<ov::op::v1::Reshape>(fqNode, constNode, false));
        } else {
            auto weightsNode = ov::test::utils::deprecated::make_constant(modelType, weightsShapes, weightsData,
                                                                          weightsData.empty());
            weights = weightsNode;
        }

        auto groupConv = std::dynamic_pointer_cast<ov::op::v1::GroupConvolution>(
                make_group_convolution(dataFq, weights, modelType, stride, padBegin, padEnd, dilation, padType));

        const auto outFq =
                ov::test::utils::make_fake_quantize(groupConv, modelType, quantLevels, {}, {0}, {255}, {0}, {255});

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(outFq)};
        VpuOv2LayerTest::function = std::make_shared<ov::Model>(results, params, "QuantGroupConvolution");
    }
    void TearDown() override {
        VpuOv2LayerTest::TearDown();
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<quantGroupConvLayerTestParamsSet>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << QuantGroupConvLayerTest::getTestCaseName(obj) << sep;

        return result.str();
    }
};

TEST_P(QuantGroupConvSubGraphTest_NPU3700, SW) {
    VpuOv2LayerTest::setReferenceSoftwareMode();
    VpuOv2LayerTest::run(Platform::NPU3700);
}

TEST_P(QuantGroupConvSubGraphTest_NPU3700, HW) {
    VpuOv2LayerTest::setDefaultHardwareMode();
    VpuOv2LayerTest::run(Platform::NPU3700);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const std::vector<size_t> numOutChannels = {3, 24, 48};
const std::vector<size_t> numGroups = {3};

const std::vector<size_t> levels = {256};
const std::vector<QuantizationGranularity> granularity = {QuantizationGranularity::Pertensor,
                                                          QuantizationGranularity::Perchannel};
const std::vector<bool> quantizeWeights2D = {true};

/* ============= 2D GroupConvolution ============= */
const std::vector<ov::Shape> inputShapes2D = {{1, 3, 10, 10}, {1, 24, 10, 10}};
const std::vector<ov::Shape> kernels2D = {{1, 1}, {3, 3}};
const std::vector<ov::Shape> strides2D = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {{0, 0}};
const std::vector<ov::Shape> dilations2D = {{1, 1}};

const auto quantGroupConv2DParams = ::testing::Combine(
        ::testing::ValuesIn(kernels2D), ::testing::ValuesIn(strides2D), ::testing::ValuesIn(padBegins2D),
        ::testing::ValuesIn(padEnds2D), ::testing::ValuesIn(dilations2D), ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups), ::testing::ValuesIn(levels), ::testing::ValuesIn(granularity),
        ::testing::ValuesIn(quantizeWeights2D));

INSTANTIATE_TEST_SUITE_P(smoke_QuantGroupConv2D, QuantGroupConvSubGraphTest_NPU3700,
                         ::testing::Combine(quantGroupConv2DParams, ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(inputShapes2D), ::testing::Values(DEVICE_NPU)),
                         QuantGroupConvSubGraphTest_NPU3700::getTestCaseName);

/* ============= 3D GroupConvolution ============= */
const std::vector<ov::Shape> inputShapes3D = {{1, 3, 5, 5, 5}, {1, 24, 5, 5, 5}};
const std::vector<ov::Shape> kernels3D = {{3, 3, 3}};
const std::vector<ov::Shape> strides3D = {{1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins3D = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds3D = {{0, 0, 0}};
const std::vector<ov::Shape> dilations3D = {{1, 1, 1}};
const std::vector<bool> quantizeWeights3D = {true};

const auto quantGroupConv3DParams = ::testing::Combine(
        ::testing::ValuesIn(kernels3D), ::testing::ValuesIn(strides3D), ::testing::ValuesIn(padBegins3D),
        ::testing::ValuesIn(padEnds3D), ::testing::ValuesIn(dilations3D), ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups), ::testing::ValuesIn(levels), ::testing::ValuesIn(granularity),
        ::testing::ValuesIn(quantizeWeights3D));

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_QuantGroupConv3D, QuantGroupConvSubGraphTest_NPU3700,
                         ::testing::Combine(quantGroupConv3DParams, ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(inputShapes3D), ::testing::Values(DEVICE_NPU)),
                         QuantGroupConvSubGraphTest_NPU3700::getTestCaseName);

}  // namespace
