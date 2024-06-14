// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/scatter_elements_update.hpp"
#include <vector>

#include "vpu_ov2_layer_test.hpp"

namespace ov {

namespace test {

class ScatterElementsUpdateLayerTestCommon : public ScatterElementsUpdateLayerTest, virtual public VpuOv2LayerTest {};

class ScatterElementsUpdateLayerTest_NPU3720 : public ScatterElementsUpdateLayerTestCommon {};
class ScatterElementsUpdateLayerTest_NPU4000 : public ScatterElementsUpdateLayerTestCommon {};

TEST_P(ScatterElementsUpdateLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ScatterElementsUpdateLayerTest_NPU4000, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU4000);
}

}  // namespace test

}  // namespace ov

using ov::test::ScatterElementsUpdateLayerTest_NPU3720;
using ov::test::ScatterElementsUpdateLayerTest_NPU4000;

namespace {

std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> axesShapeInShape{
        {{2, 3, 4}, {{{1, 3, 1}, {1, -1}}}}};

const std::vector<std::vector<size_t>> indicesValue = {{1, 0, 1}};

std::vector<ov::test::axisShapeInShape> combineShapes(
        const std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>>& input_shapes) {
    std::vector<ov::test::axisShapeInShape> res_vec;
    for (auto& input_shape : input_shapes) {
        for (auto& item : input_shape.second) {
            for (auto& elt : item.second) {
                res_vec.push_back(ov::test::axisShapeInShape{
                        ov::test::static_shapes_to_test_representation({input_shape.first, item.first}), elt});
            }
        }
    }
    return res_vec;
}

INSTANTIATE_TEST_SUITE_P(smoke_ScatterElementsUpdate, ScatterElementsUpdateLayerTest_NPU3720,
                         testing::Combine(testing::ValuesIn(combineShapes(axesShapeInShape)),
                                          testing::ValuesIn(indicesValue), testing::Values(ov::element::f16),
                                          testing::Values(ov::element::i32),
                                          testing::Values(ov::test::utils::DEVICE_NPU)),
                         ScatterElementsUpdateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ScatterElementsUpdate, ScatterElementsUpdateLayerTest_NPU4000,
                         testing::Combine(testing::ValuesIn(combineShapes(axesShapeInShape)),
                                          testing::ValuesIn(indicesValue), testing::Values(ov::element::f16),
                                          testing::Values(ov::element::i32),
                                          testing::Values(ov::test::utils::DEVICE_NPU)),
                         ScatterElementsUpdateLayerTest_NPU4000::getTestCaseName);

}  // namespace
