// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <common/functions.h>
#include <common_test_utils/ov_tensor_utils.hpp>
#include "single_op_tests/adaptive_pooling.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class AdaPoolLayerTestCommon : public AdaPoolLayerTest, virtual public VpuOv2LayerTest {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 8, 0, 32);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }
};

TEST_P(AdaPoolLayerTestCommon, NPU3720_SW) {
    abs_threshold = 0.02;
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(AdaPoolLayerTestCommon, NPU4000_SW) {
    abs_threshold = 0.02;
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f16,
        ov::element::f32,

};

std::vector<std::vector<ov::Shape>> inShape3DCases = {{{2, 3, 7}}, {{1, 1, 3}}};
std::vector<std::vector<ov::Shape>> inShape4DCases = {{{1, 3, 32, 32}}, {{1, 1, 3, 2}}};
std::vector<std::vector<ov::Shape>> inShape5DCases = {{{1, 17, 4, 5, 4}}, {{1, 1, 3, 2, 3}}};

/* ============= 3D/4D AdaptivePool ============= */

std::vector<std::vector<ov::Shape>> inShape3DSingleCases = {{{2, 3, 7}}};
std::vector<std::vector<ov::Shape>> inShape4DSingleCases = {{{1, 128, 32, 64}}};

const auto AdaPoolCase3D =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShape3DSingleCases)),
                           ::testing::ValuesIn(std::vector<std::vector<int>>{{1}, {3}}),
                           ::testing::ValuesIn(std::vector<std::string>{"avg", "max"}),
                           ::testing::ValuesIn(netPrecisions), ::testing::Values(DEVICE_NPU));

const auto AdaPoolCase4D =
        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShape4DSingleCases)),
                           ::testing::ValuesIn(std::vector<std::vector<int>>{{2, 2}, {3, 3}, {6, 6}}),
                           ::testing::ValuesIn(std::vector<std::string>{"avg", "max"}),
                           ::testing::ValuesIn(netPrecisions), ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_TestsAdaPool3D, AdaPoolLayerTestCommon, AdaPoolCase3D,
                         AdaPoolLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_TestsAdaPool4D, AdaPoolLayerTestCommon, AdaPoolCase4D,
                         AdaPoolLayerTestCommon::getTestCaseName);

}  // namespace
