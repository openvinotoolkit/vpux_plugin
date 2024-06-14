//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/shape_of.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {

namespace test {

class ShapeOfLayerTestCommon : public ShapeOfLayerTest, virtual public VpuOv2LayerTest {};
class ShapeOfLayerTest_NPU3700 : public ShapeOfLayerTestCommon {};
class ShapeOfLayerTest_NPU3720 : public ShapeOfLayerTestCommon {};
class ShapeOfLayerTest_NPU4000 : public ShapeOfLayerTestCommon {};

TEST_P(ShapeOfLayerTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

TEST_P(ShapeOfLayerTest_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(ShapeOfLayerTest_NPU4000, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test

}  // namespace ov

using ov::test::ShapeOfLayerTest_NPU3700;
using ov::test::ShapeOfLayerTest_NPU3720;
using ov::test::ShapeOfLayerTest_NPU4000;

namespace {
const std::vector<ov::element::Type> modelTypes = {ov::element::f16, ov::element::u8};

const std::vector<std::vector<ov::Shape>> inShapes = {
        std::vector<ov::Shape>{{10}},
        std::vector<ov::Shape>{{10, 11}},
        std::vector<ov::Shape>{{10, 11, 12}},
        std::vector<ov::Shape>{{10, 11, 12, 13}},
        std::vector<ov::Shape>{{10, 11, 12, 13, 14}},
        std::vector<ov::Shape>{{2, 3, 244, 244}},
        std::vector<ov::Shape>{{2, 4, 8, 16, 32}},
};

const std::vector<std::vector<ov::Shape>> inShapes_precommit = {
        std::vector<ov::Shape>{{3, 3, 5}},
        std::vector<ov::Shape>{{5, 7, 6, 3}},
};

const auto paramsConfig1 =
        testing::Combine(::testing::ValuesIn(modelTypes), ::testing::Values(ov::element::i32),
                         ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)),
                         ::testing::Values(ov::test::utils::DEVICE_NPU));
const auto paramsPrecommit =
        testing::Combine(::testing::Values(ov::element::f16), ::testing::Values(ov::element::i32),
                         ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes_precommit)),
                         ::testing::Values(ov::test::utils::DEVICE_NPU));

// --------- NPU3700 ---------
// Tracking number [E#85137]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Check, ShapeOfLayerTest_NPU3700,
                         ::testing::Combine(::testing::ValuesIn(modelTypes), ::testing::Values(ov::element::i64),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                    std::vector<std::vector<ov::Shape>>({{{10, 10, 10}}}))),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU)),
                         ShapeOfLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf, ShapeOfLayerTest_NPU3700, paramsConfig1,
                         ShapeOfLayerTest_NPU3700::getTestCaseName);

// --------- NPU3720 ---------

INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf, ShapeOfLayerTest_NPU3720, paramsConfig1,
                         ShapeOfLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ShapeOf, ShapeOfLayerTest_NPU3720, paramsPrecommit,
                         ShapeOfLayerTest_NPU3720::getTestCaseName);

// --------- NPU4000 ---------
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf, ShapeOfLayerTest_NPU4000, paramsConfig1,
                         ShapeOfLayerTest_NPU4000::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ShapeOf, ShapeOfLayerTest_NPU4000, paramsPrecommit,
                         ShapeOfLayerTest_NPU4000::getTestCaseName);

}  // namespace
