// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/scatter_update.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {

namespace test {

class ScatterUpdateLayerTestCommon : public ScatterUpdateLayerTest, virtual public VpuOv2LayerTest {};

class ScatterUpdateLayerTest_NPU3700 : public ScatterUpdateLayerTestCommon {};
class ScatterUpdateLayerTest_NPU3720 : public ScatterUpdateLayerTestCommon {};
class ScatterUpdateLayerTest_NPU4000 : public ScatterUpdateLayerTestCommon {};

TEST_P(ScatterUpdateLayerTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

TEST_P(ScatterUpdateLayerTest_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ScatterUpdateLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(ScatterUpdateLayerTest_NPU4000, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test

}  // namespace ov

using ov::test::ScatterUpdateLayerTest_NPU3700;
using ov::test::ScatterUpdateLayerTest_NPU3720;
using ov::test::ScatterUpdateLayerTest_NPU4000;

namespace {
// map<inputShape, map<indicesShape, axis>>
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> axesShapeInShape{
        {{10, 16, 12, 15}, {{{8}, {0, -2}}}}};

std::vector<ov::test::axisUpdateShapeInShape> combineShapes(
        const std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>>& input_shapes) {
    std::vector<ov::test::axisUpdateShapeInShape> res_vec;
    for (auto& input_shape : input_shapes) {
        auto src_shape = input_shape.first;
        auto srcRank = src_shape.size();
        for (auto& item : input_shape.second) {
            auto indices_shape = item.first;
            auto indices_rank = indices_shape.size();
            for (auto& axis : item.second) {
                auto axisP = axis < 0 ? axis + srcRank : axis;
                std::vector<size_t> update_shape;
                for (size_t rs = 0; rs < srcRank; rs++) {
                    if (rs != axisP) {
                        update_shape.push_back(src_shape[rs]);
                    } else {
                        for (size_t ri = 0; ri < indices_rank; ri++) {
                            update_shape.push_back(indices_shape[ri]);
                        }
                    }
                }
                std::vector<ov::Shape> in_shapes{src_shape, update_shape};
                res_vec.push_back(ov::test::axisUpdateShapeInShape{
                        ov::test::static_shapes_to_test_representation(in_shapes), ov::Shape{indices_shape}, axis});
            }
        }
    }
    return res_vec;
}

const std::vector<std::vector<int64_t>> scatterIndices = {{0, 2, 4, 6, 1, 3, 5, 7}};
const auto params = testing::Combine(testing::ValuesIn(combineShapes(axesShapeInShape)),
                                     testing::ValuesIn(scatterIndices), testing::Values(ov::element::f16),
                                     testing::Values(ov::element::i32), testing::Values(ov::test::utils::DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_ScatterUpdate, ScatterUpdateLayerTest_NPU3700, params,
                         ScatterUpdateLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ScatterUpdate, ScatterUpdateLayerTest_NPU3720, params,
                         ScatterUpdateLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ScatterUpdate, ScatterUpdateLayerTest_NPU4000, params,
                         ScatterUpdateLayerTest_NPU4000::getTestCaseName);

}  // namespace
