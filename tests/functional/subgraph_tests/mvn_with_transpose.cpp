// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu_ov2_layer_test.hpp"

using namespace ov::test;

namespace {

class MVNWithTransposeTest_NPU3720 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<std::tuple<std::vector<int64_t>, ov::Layout>> {
    void SetUp() override {
        const auto transposeOrder = std::get<0>(GetParam());
        ov::Layout outLayout = std::get<1>(GetParam());
        const ov::Shape lhsInputShape = {1, 3, 32, 64};
        inType = outType = ov::element::f16;

        init_input_shapes(static_shapes_to_test_representation({lhsInputShape}));

        const ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        const auto add = buildMVN(params.at(0));
        const auto transpose = buildTranspose(add->output(0), transposeOrder);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(transpose)};

        function = std::make_shared<ov::Model>(results, params, "MVNWithTransposeTest");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout("NHWC");
        preProc.input().model().set_layout("NHWC");
        preProc.output().tensor().set_layout(outLayout);
        preProc.output().model().set_layout(outLayout);
        function = preProc.build();
        rel_threshold = 0.5f;
    }

    std::shared_ptr<ov::Node> buildMVN(const ov::Output<ov::Node>& param) {
        auto mvn_node = std::make_shared<ov::op::v0::MVN>(param, false, true, 9.9999997473787516E-6);
        return mvn_node;
    }

    std::shared_ptr<ov::Node> buildTranspose(const ov::Output<ov::Node>& param, const std::vector<int64_t>& dimsOrder) {
        auto order = ov::op::v0::Constant::create(ov::element::i64, {dimsOrder.size()}, dimsOrder);
        return std::make_shared<ov::op::v1::Transpose>(param, order);
    }

public:
    static std::string getTestCaseName(
            const testing::TestParamInfo<std::tuple<std::vector<int64_t>, ov::Layout>>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_P(MVNWithTransposeTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

const std::vector<std::vector<int64_t>> transposes = {
        {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1},
};

const std::vector<ov::Layout> outLayout = {ov::Layout("NCHW"), ov::Layout("NHWC")};

INSTANTIATE_TEST_SUITE_P(smoke_mvntranspose, MVNWithTransposeTest_NPU3720,
                         ::testing::Combine(::testing::ValuesIn(transposes), ::testing::ValuesIn(outLayout)),
                         MVNWithTransposeTest_NPU3720::getTestCaseName);

}  // namespace
