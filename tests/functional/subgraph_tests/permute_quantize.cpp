//
// Copyright (C) 2023 - 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/nce_tasks.hpp"

namespace ov::test {

class PermuteQuantizeTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<
                std::tuple<ov::Shape, ov::element::Type, ov::element::Type, ov::Layout, ov::Layout>> {
    void SetUp() override {
        ov::Shape inputShape;
        ov::Layout inLayout, outLayout;
        std::tie(inputShape, inType, outType, inLayout, outLayout) = GetParam();

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(inType, inputDynamicShapes.front())};

        const auto nceTask = NCETasksHelpers::buildNCETask(params.at(0), NCETasksHelpers::NCEOpType::GroupConv2d);
        const auto quantRange = std::array<float, 4>{0.f, 255.f, 0.f, 255.f};
        const auto quantOp = NCETasksHelpers::quantize(nceTask, quantRange);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(quantOp)};

        function = std::make_shared<ov::Model>(results, params, "PermuteQuantizeTest");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(inLayout);
        preProc.input().model().set_layout(inLayout);
        preProc.output().tensor().set_layout(outLayout);
        preProc.output().model().set_layout(outLayout);
        function = preProc.build();
        rel_threshold = 0.5f;
    }

public:
    static std::string getTestCaseName(
            const testing::TestParamInfo<
                    std::tuple<ov::Shape, ov::element::Type, ov::element::Type, ov::Layout, ov::Layout>>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

class PermuteQuantizeTest_NPU3720 : public PermuteQuantizeTestCommon {};
class PermuteQuantizeTest_NPU4000 : public PermuteQuantizeTestCommon {};

TEST_P(PermuteQuantizeTest_NPU3720, HW) {
    setDefaultHardwareMode();
    configuration["PERFORMANCE_HINT"] = "LATENCY";
    configuration["NPU_TILES"] = "2";
    run(Platform::NPU3720);
}

TEST_P(PermuteQuantizeTest_NPU4000, HW) {
    setDefaultHardwareMode();
    configuration["PERFORMANCE_HINT"] = "THROUGHPUT";
    configuration["NPU_TILES"] = "1";
    configuration["NPU_DMA_ENGINES"] = "1";
    run(Platform::NPU4000);
}

const std::vector<ov::Shape> inputShapes = {
        {1, 3, 224, 224}, {1, 3, 128, 256}, {1, 1, 64, 64}, {1, 3, 224, 225}, {1, 3, 640, 320},
};

INSTANTIATE_TEST_SUITE_P(smoke_PermuteQuantizeTest, PermuteQuantizeTest_NPU3720,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f16), ::testing::Values(ov::Layout("NCHW")),
                                            ::testing::Values(ov::Layout("NHWC"))),
                         PermuteQuantizeTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_PermuteQuantizeTest, PermuteQuantizeTest_NPU4000,
                         ::testing::Combine(::testing::ValuesIn(inputShapes), ::testing::Values(ov::element::f16),
                                            ::testing::Values(ov::element::f16), ::testing::Values(ov::Layout("NCHW")),
                                            ::testing::Values(ov::Layout("NHWC"))),
                         PermuteQuantizeTest_NPU4000::getTestCaseName);

}  // namespace ov::test
