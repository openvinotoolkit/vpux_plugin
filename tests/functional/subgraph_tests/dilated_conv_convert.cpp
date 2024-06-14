//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpu_ov2_layer_test.hpp>

namespace ov::test {

// This test aims for:
//   - Check DilatedConvConvertPass has right implemented
// From:
//       [input]
//          |
//    (SpaceToBatch)
//          |
//         (FQ)
//          |
//  (conv/grop conv) --- (FQ) -- [filter]
//          |
//    (BatchToSpace)
//          |
//       [output]
// To:
//       [input]
//          |
//         (FQ)
//          |
// (new conv/grop conv) --- (FQ) -- [filter]
//          |
//       [output]

using DilatedConvConvertTestParams = std::tuple<bool>;

class DilatedConvConvertSubGraphTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<DilatedConvConvertTestParams> {
    void SetUp() override {
        auto isConvOp = std::get<bool>(GetParam());

        const ov::Shape inputShape{1, 16, 10, 10};
        init_input_shapes({ov::test::InputShape{{}, std::vector<ov::Shape>{inputShape}}});
        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        // Create spaceToBatch
        const auto spaceToBatch = std::make_shared<ov::op::v1::SpaceToBatch>(
                params[0], op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                op::v0::Constant::create(ov::element::i64, Shape{4}, {0, 0, 2, 2}),
                op::v0::Constant::create(ov::element::i64, Shape{4}, {0, 0, 2, 2}));

        // Create weights
        auto weightsShape = isConvOp ? ov::Shape{16, 16, 3, 3} : ov::Shape{16, 1, 1, 3, 3};
        const auto weightsSize =
                std::accumulate(weightsShape.cbegin(), weightsShape.cend(), 1, std::multiplies<size_t>());
        std::vector<float16> weightsVal(weightsSize, 0.15f);
        const auto weights = ov::op::v0::Constant::create(ov::element::f32, weightsShape, weightsVal);

        // create Conv
        const ov::Strides strides = {1, 1};
        const ov::CoordinateDiff pads_begin = {0, 0};
        const ov::CoordinateDiff pads_end = {0, 0};
        const ov::Strides dilations = {1, 1};
        std::shared_ptr<Node> nceOp;
        if (isConvOp) {
            nceOp = std::make_shared<ov::op::v1::Convolution>(spaceToBatch, weights, strides, pads_begin, pads_end,
                                                              dilations);
        } else {
            nceOp = std::make_shared<ov::op::v1::GroupConvolution>(spaceToBatch, weights, strides, pads_begin, pads_end,
                                                                   dilations);
        }

        // Create batchToSpace
        const auto batchToSpace = std::make_shared<ov::op::v1::BatchToSpace>(
                nceOp, op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 2, 2}),
                op::v0::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1}),
                op::v0::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1}));

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(batchToSpace)};
        function = std::make_shared<ov::Model>(results, params, "DilatedConvConvert");
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<DilatedConvConvertTestParams> obj) {
        auto isConvOp = std::get<bool>(obj.param);

        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        if (isConvOp) {
            result << "convolution";
        } else {
            result << "group_convolution";
        }
        return result.str();
    }
};

class DilatedConvConvertSubGraphTest_NPU3720 : public DilatedConvConvertSubGraphTestCommon {};

class DilatedConvConvertSubGraphTest_NPU4000 : public DilatedConvConvertSubGraphTestCommon {};

TEST_P(DilatedConvConvertSubGraphTest_NPU3720, HW) {
    rel_threshold = 0.1f;
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(DilatedConvConvertSubGraphTest_NPU4000, HW) {
    rel_threshold = 0.1f;
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

const auto basicCases =
        ::testing::Combine(::testing::ValuesIn(std::vector<bool>{/*isConvOp=*/true, /*isConvOp=*/false}));

INSTANTIATE_TEST_SUITE_P(precommit_DilatedConvConvert, DilatedConvConvertSubGraphTest_NPU3720, basicCases,
                         DilatedConvConvertSubGraphTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(precommit_DilatedConvConvert, DilatedConvConvertSubGraphTest_NPU4000, basicCases,
                         DilatedConvConvertSubGraphTestCommon::getTestCaseName);

}  // namespace ov::test
