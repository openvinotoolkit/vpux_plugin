// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

namespace ov::test {

struct GroupNormalizationDecomposeTestParams {
    ov::Shape dataShape;
    ov::Shape scaleShape;
    ov::Shape biasShape;
    int64_t numGroups;
};

class GroupNormalizationDecomposeTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<GroupNormalizationDecomposeTestParams> {
    void SetUp() override {
        const auto testParams = GetParam();
        const auto dataShape = testParams.dataShape;
        const auto scaleShape = testParams.scaleShape;
        const auto biasShape = testParams.biasShape;
        const auto numGroups = testParams.numGroups;
        init_input_shapes(ov::test::static_shapes_to_test_representation({dataShape, scaleShape, biasShape}));

        ov::ParameterVector params;
        for (const auto& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape));
        }

        const auto group_norm = std::make_shared<ov::op::v12::GroupNormalization>(params[0], params[1], params[2],
                                                                                  numGroups, /*epsVal=*/1e-6f);

        const ov::ResultVector outputs{std::make_shared<ov::op::v0::Result>(group_norm)};
        function = std::make_shared<ov::Model>(outputs, params, "GroupNormalizationDecomposeTest");
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<GroupNormalizationDecomposeTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_P(GroupNormalizationDecomposeTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(GroupNormalizationDecomposeTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

INSTANTIATE_TEST_SUITE_P(smoke_GroupNormalizationDecompose, GroupNormalizationDecomposeTestCommon,
                         ::testing::Values(GroupNormalizationDecomposeTestParams{{1, 160, 32, 32}, {160}, {160}, 16}),
                         GroupNormalizationDecomposeTestCommon::getTestCaseName);

}  // namespace ov::test
