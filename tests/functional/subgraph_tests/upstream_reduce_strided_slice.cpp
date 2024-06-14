//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

#include <random>

namespace ov::test {

struct UpstreamReduceStridedSliceTestParams {
    std::vector<ov::Shape> _inDims;
    std::vector<ov::Shape> _filterDims;
    std::vector<std::vector<size_t>> _kernelStrides;
    std::vector<std::vector<int64_t>> _beginData;
    std::vector<std::vector<int64_t>> _endData;
    std::vector<std::vector<int64_t>> _beginMask;
    std::vector<std::vector<int64_t>> _endMask;
};

// Build following graph which exercises the possibility to
// upstream Slice_1 and later on fuse it with Slice_0
//
// ...  Conv --- > Slice_0 (H) --- >        < --- Reshape ...
//                                 |       |
//                                 v       v
//                                 Eltwise_0
//                                     |
//                                     V
//                                 Slice_1 (H)  --- > ...

class UpstreamReduceStridedSliceSubGraphTest_NPU3700 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<UpstreamReduceStridedSliceTestParams> {
    void ValidateTestParams(UpstreamReduceStridedSliceTestParams testParams) {
        ASSERT_EQ(testParams._inDims.size(), 2);
        ASSERT_EQ(testParams._filterDims.size(), 2);
        ASSERT_EQ(testParams._kernelStrides.size(), 2);
        ASSERT_EQ(testParams._beginData.size(), 2);
        ASSERT_EQ(testParams._endData.size(), 2);
        ASSERT_EQ(testParams._beginMask.size(), 2);
        ASSERT_EQ(testParams._endMask.size(), 2);
    }

    void SetUp() override {
        const auto testParams = GetParam();
        ValidateTestParams(testParams);

        const auto inputShapeArray = testParams._inDims;
        init_input_shapes(static_shapes_to_test_representation(inputShapeArray));

        ov::ParameterVector params;
        for (const auto& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape));
        }

        std::mt19937 floatMersenneEngine{0};
        std::normal_distribution<double> floatDist{0.0, 1.0};
        auto floatGen = [&floatDist, &floatMersenneEngine]() {
            return floatDist(floatMersenneEngine);
        };

        std::mt19937 intMersenneEngine{0};
        std::normal_distribution<double> intDist{-127.0, 127.0};
        auto intGen = [&intDist, &intMersenneEngine]() {
            return (int8_t)std::round(intDist(intMersenneEngine));
        };

        std::vector<std::shared_ptr<ov::op::Op>> nodes;

        for (size_t idx = 0; idx < params.size(); idx++) {
            const auto filterShape = testParams._filterDims[idx];
            std::vector<int8_t> weightsData(ov::shape_size(filterShape));
            std::generate(weightsData.begin(), weightsData.end(), intGen);
            const auto weightsConst = std::make_shared<ov::op::v0::Constant>(ov::element::i8, filterShape, weightsData);

            const auto weightsConvert = std::make_shared<ov::op::v0::Convert>(weightsConst, ov::element::f32);

            std::vector<double> subtractData(filterShape[0], 0);
            const auto subtractConst = std::make_shared<ov::op::v0::Constant>(
                    ov::element::f32, ov::Shape({filterShape[0], 1, 1, 1}), subtractData);
            const auto weightsSubtract = std::make_shared<ov::op::v1::Subtract>(weightsConvert, subtractConst);

            std::vector<double> multiplyData(filterShape[0], 0.00784313725490196);
            const auto multiplyConst = std::make_shared<ov::op::v0::Constant>(
                    ov::element::f32, ov::Shape({filterShape[0], 1, 1, 1}), multiplyData);
            const auto weightsMultiply = std::make_shared<ov::op::v1::Multiply>(weightsSubtract, multiplyConst);

            const auto inputLowConst = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape({1}),
                                                                              std::vector<double>({-1.0}));
            const auto inputHighConst = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape({1}),
                                                                               std::vector<double>({1.0}));
            const auto inputFq = std::make_shared<ov::op::v0::FakeQuantize>(params[idx], inputLowConst, inputHighConst,
                                                                            inputLowConst, inputHighConst, 256);

            const auto conv = std::make_shared<ov::op::v1::Convolution>(
                    inputFq, weightsMultiply, ov::Strides(testParams._kernelStrides[idx]),
                    ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                    ov::Strides(testParams._kernelStrides[idx]), ov::op::PadType::AUTO);

            const auto biasShape = ov::Shape({1, conv->get_output_shape(0)[1], 1, 1});
            std::vector<double> biasData(ov::shape_size(biasShape));
            std::generate(biasData.begin(), biasData.end(), floatGen);
            const auto biasConst = std::make_shared<ov::op::v0::Constant>(ov::element::f32, biasShape, biasData);
            const auto bias = std::make_shared<ov::op::v1::Add>(conv, biasConst);

            const auto relu = std::make_shared<ov::op::v0::Relu>(bias);

            const auto actLowConst = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape({1}),
                                                                            std::vector<double>({0.0}));
            const auto actHighConst = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape({1}),
                                                                             std::vector<double>({4.0}));
            const auto activationFq = std::make_shared<ov::op::v0::FakeQuantize>(relu, actLowConst, actHighConst,
                                                                                 actLowConst, actHighConst, 256);

            nodes.push_back(activationFq);
        }

        const auto convSliceBeginConst = ov::op::v0::Constant::create(
                ov::element::i64, ov::Shape{testParams._beginData[0].size()}, testParams._beginData[0]);
        const auto convSliceEndConst = ov::op::v0::Constant::create(
                ov::element::i64, ov::Shape{testParams._endData[0].size()}, testParams._endData[0]);
        const auto convSliceStridesConst = ov::op::v0::Constant::create(
                ov::element::i64, ov::Shape{testParams._beginData[0].size()}, {1, 1, 1, 1});

        const auto convStridedSlice = std::make_shared<ov::op::v1::StridedSlice>(
                nodes[0], convSliceBeginConst, convSliceEndConst, convSliceStridesConst, testParams._beginMask[0],
                testParams._endMask[0]);

        // Adjust shape by doubling width and halfing channels
        const auto reshapePattern =
                std::vector<int64_t>({0, -1, 0, static_cast<int64_t>(2 * nodes[1]->get_output_shape(0)[3])});
        const auto newShape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, reshapePattern);
        const auto reshape = std::make_shared<ov::op::v1::Reshape>(nodes[1], newShape, true);

        const auto add = std::make_shared<ov::op::v1::Add>(convStridedSlice, reshape);

        const auto relu = std::make_shared<ov::op::v0::Relu>(add);

        const auto resultLowConst =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape({1}), std::vector<double>({0.0}));
        const auto resultHighConst =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape({1}), std::vector<double>({8.0}));

        const auto reluFq = std::make_shared<ov::op::v0::FakeQuantize>(relu, resultLowConst, resultHighConst,
                                                                       resultLowConst, resultHighConst, 256);

        const auto addSliceBeginConst = ov::op::v0::Constant::create(
                ov::element::i64, ov::Shape{testParams._beginData[1].size()}, testParams._beginData[1]);
        const auto addSliceEndConst = ov::op::v0::Constant::create(
                ov::element::i64, ov::Shape{testParams._endData[1].size()}, testParams._endData[1]);
        const auto addSliceStridesConst = ov::op::v0::Constant::create(
                ov::element::i64, ov::Shape{testParams._beginData[1].size()}, {1, 1, 1, 1});

        const auto addStridedSlice = std::make_shared<ov::op::v1::StridedSlice>(
                reluFq, addSliceBeginConst, addSliceEndConst, addSliceStridesConst, testParams._beginMask[1],
                testParams._endMask[1]);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(addStridedSlice)};

        function = std::make_shared<ov::Model>(results, params, "UpstreamReduceStridedSliceSubGraphTest");
        rel_threshold = 0.5f;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<UpstreamReduceStridedSliceTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_P(UpstreamReduceStridedSliceSubGraphTest_NPU3700, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3700);
}

TEST_P(UpstreamReduceStridedSliceSubGraphTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

INSTANTIATE_TEST_CASE_P(smoke_UpstreamReduceStridedSlice, UpstreamReduceStridedSliceSubGraphTest_NPU3700,
                        ::testing::Values(UpstreamReduceStridedSliceTestParams{
                                {{1, 16, 64, 4}, {1, 16, 17, 2}},  // in dims
                                {{32, 16, 3, 1}, {64, 16, 3, 1}},  // filter dims
                                {{1, 1}, {1, 1}},                  // kernel strides
                                {{0, 0, 47, 0}, {0, 0, 1, 0}},     // begin data
                                {{0, 0, 0, 0}, {0, 0, 0, 0}},      // end data
                                {{1, 1, 0, 1}, {1, 1, 0, 1}},      // begin mask
                                {{1, 1, 1, 1}, {1, 1, 1, 1}}       // end mask
                        }),
                        UpstreamReduceStridedSliceSubGraphTest_NPU3700::getTestCaseName);

}  // namespace ov::test
