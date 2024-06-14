//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>
#include "common/functions.h"

#include "common_test_utils/node_builders/constant.hpp"

namespace ov::test {

typedef std::tuple<ov::element::Type, ov::element::Type, ov::Shape, ov::Shape, ov::Strides, ov::Strides,
                   std::pair<ov::CoordinateDiff, ov::CoordinateDiff>, ov::Layout>
        CMajorConvNHWCTestParams;
class CMajorConvNHWCTest_NPU3700 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<CMajorConvNHWCTestParams> {
    void SetUp() override {
        auto prms = GetParam();

        ov::Shape inputShape;
        ov::Shape weightsShape;
        ov::Strides strides;
        std::pair<ov::CoordinateDiff, ov::CoordinateDiff> pads;
        ov::Strides dilations;
        ov::Layout order;

        std::tie(inType, outType, inputShape, weightsShape, strides, dilations, pads, order) = prms;

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        const auto weightsU8 =
                ov::test::utils::deprecated::make_constant<uint8_t>(ov::element::u8, weightsShape, {}, true, 255, 0);
        const auto weightsFP32 = std::make_shared<ov::op::v0::Convert>(weightsU8, ov::element::f32);

        const auto conv = std::make_shared<ov::op::v1::Convolution>(params.at(0), weightsFP32, strides, pads.first,
                                                                    pads.second, dilations);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(conv)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{params}, "CMajorConvNHWCTest");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(order);
        preProc.input().model().set_layout(order);
        preProc.output().tensor().set_layout(order);
        preProc.output().model().set_layout(order);
        function = preProc.build();
        rel_threshold = 0.1f;
    }

    template <typename T>
    static std::string VectorToString(std::vector<T> v) {
        std::ostringstream res;
        for (size_t i = 0; i < v.size(); ++i) {
            if (i != 0) {
                res << ",";
            } else {
                res << "{";
            }

            res << v[i];
        }
        res << "}";
        return res.str();
    }

public:
    static std::string getTestCaseName(::testing::TestParamInfo<CMajorConvNHWCTestParams> obj) {
        auto params = obj.param;

        ov::element::Type ip, op;
        ov::Shape inputShape;
        ov::Shape weightsShape;
        ov::Strides strides;
        std::pair<ov::CoordinateDiff, ov::CoordinateDiff> pads;
        ov::Strides dilations;

        std::tie(ip, op, inputShape, weightsShape, strides, dilations, pads, std::ignore) = params;

        const std::string sep = "_";
        std::ostringstream result;

        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "InputPrec=" << ip << sep;
        result << "OutputPrec=" << op << sep;
        result << "InShape=" << VectorToString(inputShape) << sep;
        result << "WeightsShape=" << VectorToString(weightsShape) << sep;
        result << "Strides=" << VectorToString(strides) << sep;
        result << "Dilations=" << VectorToString(dilations) << sep;
        result << "Padding=" << VectorToString(pads.first) << "," << VectorToString(pads.second);
        return result.str();
    }
};

TEST_P(CMajorConvNHWCTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

const std::vector<ov::element::Type> prec = {ov::element::f16, ov::element::f32, ov::element::u8};

const std::vector<ov::Shape> inputShapes{{1, 3, 32, 32}};

const std::vector<ov::Shape> weightShapes{{16, 3, 1, 1}};

const std::vector<ov::Strides> strides{{1, 1}};

const std::vector<ov::Strides> dilations{{1, 1}};

const std::vector<std::pair<ov::CoordinateDiff, ov::CoordinateDiff>> pads{{{0, 0}, {0, 0}}};

/* NOTE: Tests have not yet run on actual device, because of CI instability. Only LoadNetwork phase was done. */
INSTANTIATE_TEST_CASE_P(smoke_Permute_NHWC_To_NCHW, CMajorConvNHWCTest_NPU3700,
                        ::testing::Combine(::testing::ValuesIn(prec), ::testing::ValuesIn(prec),
                                           ::testing::ValuesIn(inputShapes), ::testing::ValuesIn(weightShapes),
                                           ::testing::ValuesIn(strides), ::testing::ValuesIn(dilations),
                                           ::testing::ValuesIn(pads), ::testing::Values(ov::Layout("NHWC"))),
                        CMajorConvNHWCTest_NPU3700::getTestCaseName);

}  // namespace ov::test
