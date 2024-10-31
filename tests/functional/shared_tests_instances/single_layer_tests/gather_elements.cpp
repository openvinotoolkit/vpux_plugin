// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <common/functions.h>
#include <common_test_utils/ov_tensor_utils.hpp>
#include "single_op_tests/gather_elements.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {
class GatherElementsLayerTestCommon : public GatherElementsLayerTest, virtual public VpuOv2LayerTest {
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        VpuOv2LayerTest::inputs.clear();
        const auto& funcInputs = VpuOv2LayerTest::function->inputs();
        ov::Tensor tensorData =
                create_and_fill_tensor(funcInputs[0].get_element_type(), targetInputStaticShapes[0], 8, 0, 32);
        VpuOv2LayerTest::inputs.insert({funcInputs[0].get_node_shared_ptr(), tensorData});
    }
};

TEST_P(GatherElementsLayerTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(GatherElementsLayerTestCommon, NPU4000_HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> dPrecisions = {ov::element::f32};

const std::vector<ov::element::Type> iPrecisions = {ov::element::i32};

const std::vector<int> axes_set1 = {-1, 0, 1};
const std::vector<int> axes_set2 = {-2, 1};
const std::vector<int> axes_set3 = {0};

const std::vector<std::vector<ov::Shape>> iShapes = {{{2, 2}}, {{5, 7, 9, 1}}, {{2, 2, 1}}};

const auto GatherElements_PRECOMMIT_set1 =
        ::testing::Combine(testing::ValuesIn({static_shapes_to_test_representation(iShapes[0])}),
                           testing::Values(ov::Shape{2, 2}), testing::ValuesIn(axes_set1),
                           testing::ValuesIn(dPrecisions), testing::ValuesIn(iPrecisions), testing::Values(DEVICE_NPU));

const auto GatherElements_PRECOMMIT_set2 =
        ::testing::Combine(testing::ValuesIn({static_shapes_to_test_representation(iShapes[1])}),
                           testing::Values(ov::Shape{5, 7, 9, 1}), testing::ValuesIn(axes_set2),
                           testing::ValuesIn(dPrecisions), testing::ValuesIn(iPrecisions), testing::Values(DEVICE_NPU));

const auto GatherElements_PRECOMMIT_set3 = ::testing::Combine(
        ::testing::ValuesIn({static_shapes_to_test_representation(iShapes[2])}), ::testing::Values(ov::Shape{4, 2, 1}),
        ::testing::ValuesIn(axes_set3), ::testing::ValuesIn(dPrecisions), ::testing::ValuesIn(iPrecisions),
        ::testing::Values(DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GatherElements_set1, GatherElementsLayerTestCommon,
                         GatherElements_PRECOMMIT_set1, GatherElementsLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GatherElements_set2, GatherElementsLayerTestCommon,
                         GatherElements_PRECOMMIT_set2, GatherElementsLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GatherElements_set3, GatherElementsLayerTestCommon,
                         GatherElements_PRECOMMIT_set3, GatherElementsLayerTestCommon::getTestCaseName);

}  // namespace
