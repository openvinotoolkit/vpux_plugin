// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset1.hpp"
#include "shared_test_classes/subgraph/nce_tasks.hpp"

namespace ov::test {
struct ShapeWithNumFilters {
    ShapeWithNumFilters(const ov::Shape& shape, const size_t filters, const bool bypassQuant)
            : _inShape(shape), _outFilters(filters), _bypassQuant(bypassQuant) {
    }
    ShapeWithNumFilters(const ShapeWithNumFilters& other)
            : _inShape(other._inShape), _outFilters(other._outFilters), _bypassQuant(other._bypassQuant) {
    }
    ShapeWithNumFilters(const ShapeWithNumFilters&& other) = delete;
    ShapeWithNumFilters& operator=(const ShapeWithNumFilters& other) {
        _inShape = other._inShape;
        _outFilters = other._outFilters;
        _bypassQuant = other._bypassQuant;
        return *this;
    }
    ShapeWithNumFilters& operator=(const ShapeWithNumFilters&& other) = delete;
    ~ShapeWithNumFilters() = default;
    ov::Shape _inShape;
    size_t _outFilters;
    bool _bypassQuant;
};

class Conv2dWithExpandTest_NPU3720 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<std::tuple<ShapeWithNumFilters, ov::element::Type>> {
    ov::Output<ov::Node> quantizeValue(const ov::Output<ov::Node>& param, const bool bypassQuant,
                                       const std::array<float, 4>& quantRange, const size_t quantLevels) const {
        if (bypassQuant) {
            return param;
        }
        const auto fqOp = NCETasksHelpers::quantize(param, quantRange, quantLevels);
        return fqOp->output(0);
    }
    ov::Output<ov::Node> composeFloatWeights(const ov::Shape& weightsShape) const {
        const size_t totalSize =
                std::accumulate(weightsShape.begin(), weightsShape.end(), 1, std::multiplies<size_t>());
        std::vector<ov::float16> weights(totalSize, 0.f);
        for (size_t i = 0; i < weights.size(); i++) {
            weights.at(i) = std::cos(i * 3.14 / 6);
        }
        const auto weightsConst = ov::op::v0::Constant::create(ov::element::Type_t::f16, weightsShape, weights);
        return weightsConst->output(0);
    }
    ov::Output<ov::Node> composeQuantWeights(const ov::Shape& weightsShape) const {
        const size_t strideKX = 1;
        const size_t strideKY = strideKX * weightsShape[3];
        const size_t strideIC = strideKY * weightsShape[2];
        const size_t strideOC = strideIC * weightsShape[1];
        const size_t totalSize =
                std::accumulate(weightsShape.begin(), weightsShape.end(), 1, std::multiplies<size_t>());
        std::vector<ov::float16> weights(totalSize, 0.f);
        for (size_t oc = 0; oc < weightsShape[0]; oc++) {
            const size_t ic = oc % weightsShape[1];
            for (size_t ky = 0; ky < weightsShape[2]; ky++) {
                for (size_t kx = 0; kx < weightsShape[3]; kx++) {
                    const size_t idx = kx * strideKX + ky * strideKY + ic * strideIC + oc * strideOC;
                    weights.at(idx) = 127.f;
                }
            }
        }
        const auto weightsConst = ov::op::v0::Constant::create(ov::element::Type_t::f16, weightsShape, weights);
        const auto quantWeightsRange = std::array<float, 4>{-127.f, 127.f, -1.f, 1.f};
        const auto convWeights = quantizeValue(weightsConst->output(0), false, quantWeightsRange, 255);

        return convWeights;
    }
    ov::Output<ov::Node> composeWeights(const bool bypassQuant, const size_t outputChannels,
                                        const size_t inputChannels) const {
        const size_t kernelX = 1;
        const size_t kernelY = 1;
        const auto weightsShape = ov::Shape{outputChannels, inputChannels, kernelY, kernelX};
        if (bypassQuant) {
            return composeFloatWeights(weightsShape);
        }
        return composeQuantWeights(weightsShape);
    }
    void SetUp() override {
        const auto testParameters = std::get<0>(GetParam());
        const ov::Shape& inputShape = testParameters._inShape;
        const bool& bypassQuant = testParameters._bypassQuant;
        const size_t& inputChannels = inputShape.at(1);
        const size_t& outputChannels = testParameters._outFilters;
        inType = outType = std::get<ov::element::Type>(GetParam());
        ov::Layout order = ov::Layout("NHWC");

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        const auto quantInputRange = std::array<float, 4>{-128.f, 127.f, -128.f, 127.f};
        // need to convert param type to f16 explicitly
        const auto toF16 = std::make_shared<ov::opset1::Convert>(params.at(0), ov::element::f16);
        const auto convInput = quantizeValue(toF16->output(0), bypassQuant, quantInputRange, 256);

        const auto convWeights = composeWeights(bypassQuant, outputChannels, inputChannels);

        const auto conv = std::make_shared<ov::op::v1::Convolution>(
                convInput, convWeights, ov::Strides(std::vector<size_t>{1, 1}),
                ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                ov::Strides(std::vector<size_t>{1, 1}));

        const auto quantOutputRange = std::array<float, 4>{-128.f, 127.f, -128.f, 127.f};
        const auto convOutput = quantizeValue(conv->output(0), bypassQuant, quantOutputRange, 256);
        const auto toF32ConvOut = std::make_shared<ov::opset1::Convert>(convOutput, ov::element::f32);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(toF32ConvOut)};

        function = std::make_shared<ov::Model>(results, params, "Conv2dWithExpandTest");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(order);
        preProc.input().model().set_layout(order);
        preProc.output().tensor().set_layout(order);
        preProc.output().model().set_layout(order);
        function = preProc.build();
        rel_threshold = 0.5f;
    }

public:
    static std::string getTestCaseName(
            const testing::TestParamInfo<std::tuple<ShapeWithNumFilters, ov::element::Type>>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_P(Conv2dWithExpandTest_NPU3720, HW) {
    abs_threshold = 0.02;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

const std::vector<ShapeWithNumFilters> shapes = {
        ShapeWithNumFilters({1, 3, 112, 224}, 16, true), ShapeWithNumFilters({1, 1, 19, 80}, 16, true),
        ShapeWithNumFilters({1, 24, 64, 112}, 32, true), ShapeWithNumFilters({1, 3, 112, 224}, 16, false),
        ShapeWithNumFilters({1, 1, 19, 80}, 16, false),  ShapeWithNumFilters({1, 24, 64, 112}, 32, false),
};

INSTANTIATE_TEST_SUITE_P(conv2d_with_expand, Conv2dWithExpandTest_NPU3720,
                         ::testing::Combine(::testing::ValuesIn(shapes), ::testing::Values(ov::element::f16)),
                         Conv2dWithExpandTest_NPU3720::getTestCaseName);

}  // namespace ov::test
