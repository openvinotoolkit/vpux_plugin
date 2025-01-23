// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>

#include <gtest/gtest-param-test.h>
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

using MultipleConcatParams = std::tuple<std::vector<size_t>,  // input shape
                                        std::vector<int64_t>  // concat axes
                                        >;

class MultipleConcatTestCommon : public testing::WithParamInterface<MultipleConcatParams>, public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MultipleConcatParams> obj) {
        std::vector<size_t> inputShape;
        std::vector<int64_t> axes;
        std::tie(inputShape, axes) = obj.param;

        const std::string sep = "_";
        std::ostringstream result;
        const auto printConfigMember = [&](const std::string name, const auto& member) {
            result << name << "={";
            std::for_each(member.begin(), member.end() - 1, [&result](auto& val) {
                result << val << ", ";
            });
            result << member.back() << "}";
        };
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        printConfigMember("inputShape", inputShape);
        result << sep;
        printConfigMember("axes", axes);
        return result.str();
    }

    void SetUp() override {
        std::vector<size_t> lhsInputShapeVec;
        std::vector<int64_t> axes;
        std::tie(lhsInputShapeVec, axes) = this->GetParam();

        std::vector<size_t> rhsInputShapeVec = lhsInputShapeVec;
        InputShape lhsInputShape = {{}, std::vector<ov::Shape>({lhsInputShapeVec})};
        InputShape rhsInputShape = {{}, std::vector<ov::Shape>({rhsInputShapeVec})};
        init_input_shapes({lhsInputShape, rhsInputShape});

        auto input0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes[0]);
        input0->set_friendly_name("input_0");

        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes[1]);
        input1->set_friendly_name("input_1");

        const auto buildConcatChain = [&](ov::OutputVector inputs) {
            std::shared_ptr<ov::op::v0::Concat> concat = nullptr;
            for (auto axis : axes) {
                concat = std::make_shared<ov::op::v0::Concat>(inputs, axis);
                inputs = ov::OutputVector({concat->output(0), concat->output(0)});
            }
            return OutputVector({concat->output(0)});
        };

        const auto results = buildConcatChain(ov::OutputVector({input0, input1}));
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{input0, input1}, "MultipleConcatTest");

        rel_threshold = 0.1f;
    }
};

class MultipleConcatTest_NPU4000 : public MultipleConcatTestCommon {};

TEST_P(MultipleConcatTest_NPU4000, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace ov::test::subgraph

using namespace ov::test::subgraph;

namespace {

std::vector<std::vector<size_t>> inputSizes4D = {
        /**/ {2, 2, 2, 2},
        /**/ {3, 5, 2, 10},
        /**/};
std::vector<std::vector<int64_t>> axes_4D = {
        /**/ {1, 0, 3, 2},
        /**/ {3, 2, 3, 0},
        /**/};

std::vector<std::vector<size_t>> inputSizes5D = {
        /**/ {2, 2, 2, 2, 2},
        // /**/ {3, 12, 2, 4, 6},
        /**/};
std::vector<std::vector<int64_t>> axes_5D = {
        /**/ {2, 1, 3, 0, 4},
        /**/ {4, 1, 3, 0, 2},
        /**/};

INSTANTIATE_TEST_SUITE_P(smoke_MultipleConcatTest_4D, MultipleConcatTest_NPU4000,
                         ::testing::Combine(::testing::ValuesIn(inputSizes4D), ::testing::ValuesIn(axes_4D)),
                         MultipleConcatTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MultipleConcatTest_5D, MultipleConcatTest_NPU4000,
                         ::testing::Combine(::testing::ValuesIn(inputSizes5D), ::testing::ValuesIn(axes_5D)),
                         MultipleConcatTestCommon::getTestCaseName);

}  // namespace
