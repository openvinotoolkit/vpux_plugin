//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>
#include "common/functions.h"

namespace ov::test::subgraph {

typedef std::tuple<ov::element::Type, ov::element::Type, std::vector<ov::test::InputShape>, ov::test::TargetDevice>
        SwishLayerTestParams;
class SwishSingleInputLayerTest_NPU3700 :
        virtual public VpuOv2LayerTest,
        public testing::WithParamInterface<SwishLayerTestParams> {
    void SetUp() override {
        std::vector<ov::test::InputShape> inputShape;

        std::tie(inType, outType, inputShape, targetDevice) = GetParam();

        init_input_shapes(inputShape);

        ov::ParameterVector param;
        for (const auto& shape : inputDynamicShapes) {
            param.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f16, shape));
        }

        const auto swish = std::make_shared<ov::op::v4::Swish>(param[0]);

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(swish)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "SwishSingleInputTest");

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
    static std::string getTestCaseName(::testing::TestParamInfo<SwishLayerTestParams> obj) {
        auto params = obj.param;

        ov::element::Type ip, op;
        std::vector<ov::test::InputShape> inputShape;

        std::tie(ip, op, inputShape, std::ignore) = params;

        const std::string sep = "_";
        std::ostringstream result;

        result << "InputPrec=" << ip << sep;
        result << "OutputPrec=" << op << sep;
        result << "InShape=" << VectorToString(inputShape[0].second) << sep;

        return result.str();
    }
};

TEST_P(SwishSingleInputLayerTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}
const std::vector<ov::element::Type> netPrecision = {ov::element::undefined};

const std::vector<std::vector<ov::Shape>> inputShapes{{{1, 3, 32, 32}}, {{1, 3, 200, 200}}};

INSTANTIATE_TEST_CASE_P(
        smoke_SwishSingleInputTest, SwishSingleInputLayerTest_NPU3700,
        ::testing::Combine(::testing::ValuesIn(netPrecision), ::testing::ValuesIn(netPrecision),
                           ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes)),
                           ::testing::Values(ov::test::utils::DEVICE_NPU)),
        SwishSingleInputLayerTest_NPU3700::getTestCaseName);

}  // namespace ov::test::subgraph
