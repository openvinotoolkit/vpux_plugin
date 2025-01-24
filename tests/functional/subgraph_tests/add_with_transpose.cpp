// Copyright (C) 2023 - 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

namespace ov::test {

class AddWithTransposeTest_NPU3720 : public testing::WithParamInterface<std::vector<int64_t>>, public VpuOv2LayerTest {
    void SetUp() override {
        const auto transposeOrder = GetParam();
        const ov::Shape lhsInputShape = {1, 8, 4, 16};
        const ov::Shape rhsInputShape = {1, 8, 4, 16};

        init_input_shapes(static_shapes_to_test_representation({lhsInputShape, rhsInputShape}));

        ov::ParameterVector params;
        for (const auto& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape));
        }
        const auto lhsTranspose = buildTranspose(params.at(0), transposeOrder);
        const auto rhsTranspose = buildTranspose(params.at(1), transposeOrder);
        const auto add = buildAdd(lhsTranspose->output(0), rhsTranspose->output(0));

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add)};

        function = std::make_shared<ov::Model>(results, params, "TransposeEltwise");
        rel_threshold = 0.5f;
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

TEST_P(AddWithTransposeTest_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(AddWithTransposeTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

const std::vector<std::vector<int64_t>> transposes = {
        {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1},
};

INSTANTIATE_TEST_SUITE_P(smoke_transpose_add, AddWithTransposeTest_NPU3720, ::testing::ValuesIn(transposes),
                         AddWithTransposeTest_NPU3720::getTestCaseName);

}  // namespace ov::test
