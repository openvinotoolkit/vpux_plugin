// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

namespace ov::test {

class Conv2dWithTransposeTest_NPU3720 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<
                std::tuple<std::vector<int64_t>, ov::element::Type, ov::element::Type, ov::Layout, ov::Layout>> {
    void SetUp() override {
        std::vector<int64_t> transposeOrder;
        ov::Layout inLayout, outLayout;
        std::tie(transposeOrder, inType, outType, inLayout, outLayout) = GetParam();
        const ov::Shape lhsInputShape = {1, 3, 32, 64};

        init_input_shapes(static_shapes_to_test_representation({lhsInputShape}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        const auto add = buildConv(params.at(0));
        const auto transpose = buildTranspose(add->output(0), transposeOrder);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(transpose)};

        function = std::make_shared<ov::Model>(results, params, "Conv2dWithTransposeTest");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(inLayout);
        preProc.input().model().set_layout(inLayout);
        preProc.output().tensor().set_layout(outLayout);
        preProc.output().model().set_layout(outLayout);
        function = preProc.build();
        rel_threshold = 0.5f;
    }

    std::shared_ptr<ov::Node> buildConv(const ov::Output<ov::Node>& param) {
        const ov::Shape inputShape = param.get_shape();
        const auto weightsSize = inputShape.at(1) * 16 * 1 * 1;
        std::vector<float> values(weightsSize, 1.f);
        const auto weightsShape = ov::Shape{16, inputShape.at(1), 1, 1};
        const auto weights = ov::op::v0::Constant::create(ov::element::f32, weightsShape, values);
        auto conv2d_node = std::make_shared<ov::op::v1::Convolution>(
                param, weights->output(0), ov::Strides(std::vector<size_t>{1, 1}),
                ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                ov::Strides(std::vector<size_t>{1, 1}));

        return conv2d_node;
    }

    std::shared_ptr<ov::Node> buildTranspose(const ov::Output<ov::Node>& param, const std::vector<int64_t>& dimsOrder) {
        auto order = ov::op::v0::Constant::create(ov::element::i64, {dimsOrder.size()}, dimsOrder);
        return std::make_shared<ov::op::v1::Transpose>(param, order);
    }

public:
    static std::string getTestCaseName(
            const testing::TestParamInfo<std::tuple<std::vector<int64_t>, ov::element::Type, ov::element::Type,
                                                    ov::Layout, ov::Layout>>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_P(Conv2dWithTransposeTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

const std::vector<std::vector<int64_t>> transposes = {
        {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1}, {1, 0, 2, 3}, {2, 0, 1, 3},
        {3, 0, 1, 2}, {1, 2, 0, 3}, {2, 1, 0, 3}, {3, 1, 0, 2}, {1, 2, 3, 0}, {2, 1, 3, 0}, {3, 1, 2, 0},
};

const std::vector<ov::Layout> outLayout = {
        ov::Layout("NCHW"),
        ov::Layout("NHWC"),
};

INSTANTIATE_TEST_SUITE_P(smoke_transposeConv2d, Conv2dWithTransposeTest_NPU3720,
                         ::testing::Combine(::testing::ValuesIn(transposes), ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f16), ::testing::Values(ov::Layout("NHWC")),
                                            ::testing::ValuesIn(outLayout)),
                         Conv2dWithTransposeTest_NPU3720::getTestCaseName);

}  // namespace ov::test
