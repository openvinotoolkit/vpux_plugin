// Copyright (C) 2023 - 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

namespace ov::test {

class MatMulWithTransposeTest_NPU3720 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<std::vector<int64_t>> {
    void SetUp() override {
        const auto transposeOrder = GetParam();
        const size_t inputColumns = 64;
        const size_t outputColumns = 76;
        const ov::Shape lhsInputShape = {1, 4, 8, inputColumns};
        const ov::Shape rhsInputShape = {1, 8, 4, inputColumns};

        init_input_shapes(static_shapes_to_test_representation({lhsInputShape, rhsInputShape}));

        ov::ParameterVector params;
        for (const auto& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape));
        }
        const auto lhsGeMM = buildMatMul(params.at(0), ov::Shape{1, 4, inputColumns, outputColumns});
        const auto lhsGeMMTranspose = buildTranspose(lhsGeMM->output(0), std::vector<int64_t>{0, 2, 1, 3});
        const auto rhsGeMM = buildMatMul(params.at(1), ov::Shape{1, 8, inputColumns, outputColumns});
        const auto add = buildAdd(lhsGeMMTranspose->output(0), rhsGeMM->output(0));
        const auto transpose = buildTranspose(add->output(0), transposeOrder);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(transpose)};

        function = std::make_shared<ov::Model>(results, params, "MatMulWithTransposeTest");
        rel_threshold = 0.5f;
    }

    std::shared_ptr<ov::Node> buildMatMul(const ov::Output<ov::Node>& param, const ov::Shape& weightsShape) {
        const ov::Shape inputShape = param.get_shape();
        const auto weightsSize =
                std::accumulate(weightsShape.cbegin(), weightsShape.cend(), 1, std::multiplies<size_t>());
        std::vector<float> values(weightsSize, 1.f);

        const auto weights = ov::op::v0::Constant::create(ov::element::f32, weightsShape, values);
        return std::make_shared<ov::op::v0::MatMul>(param, weights, false, false);
    }

    std::shared_ptr<ov::Node> buildTranspose(const ov::Output<ov::Node>& param, const std::vector<int64_t>& dimsOrder) {
        auto order = ov::op::v0::Constant::create(ov::element::i64, {dimsOrder.size()}, dimsOrder);
        return std::make_shared<ov::op::v1::Transpose>(param, order);
    }

    std::shared_ptr<ov::Node> buildAdd(const ov::Output<ov::Node>& lhs, const ov::Output<ov::Node>& rhs) {
        return std::make_shared<ov::op::v1::Add>(lhs, rhs);
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::vector<int64_t>>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_P(MatMulWithTransposeTest_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(MatMulWithTransposeTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

const std::vector<std::vector<int64_t>> transposes = {
        {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1},
};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulWithTranspose, MatMulWithTransposeTest_NPU3720, ::testing::ValuesIn(transposes),
                         MatMulWithTransposeTest_NPU3720::getTestCaseName);

}  // namespace ov::test
