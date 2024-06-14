//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/batch_norm.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;
namespace ov {
namespace test {

class BatchNormLayerTestCommon : public BatchNormLayerTest, virtual public VpuOv2LayerTest {};

class BatchNormLayerTest_NPU3720 : public BatchNormLayerTestCommon {};
class BatchNormLayerTest_NPU4000 : public BatchNormLayerTestCommon {};

TEST_P(BatchNormLayerTest_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(BatchNormLayerTest_NPU4000, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<std::vector<ov::Shape>> inShapes_precommit = {
        {{1, 5, 20, 20}},
};
const std::vector<std::vector<ov::Shape>> inShapes = {
        {{6, 7}},
        {{3, 3, 5}},
        {{5, 7, 6, 3}},
        {{1, 3, 256, 256}},
};

const auto paramsConfig =
        testing::Combine(testing::Values(0.001),                                             // epsilon
                         testing::Values(ov::element::f16),                                  // Model type
                         testing::ValuesIn(static_shapes_to_test_representation(inShapes)),  // Input shape
                         testing::Values(DEVICE_NPU));                                       // Target device name

const auto paramsPrecommit =
        testing::Combine(testing::Values(0.001),                                                       // epsilon
                         testing::Values(ov::element::f16),                                            // Model type
                         testing::ValuesIn(static_shapes_to_test_representation(inShapes_precommit)),  // Input shape
                         testing::Values(DEVICE_NPU));  // Target device name

// ------ NPU3720 ------

INSTANTIATE_TEST_SUITE_P(smoke_precommit_BatchNorm, BatchNormLayerTest_NPU3720, paramsPrecommit,
                         BatchNormLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BatchNorm, BatchNormLayerTest_NPU3720, paramsConfig,
                         BatchNormLayerTest_NPU3720::getTestCaseName);

// ------ NPU4000 ------

INSTANTIATE_TEST_SUITE_P(smoke_precommit_BatchNorm, BatchNormLayerTest_NPU4000, paramsPrecommit,
                         BatchNormLayerTest_NPU4000 ::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BatchNorm, BatchNormLayerTest_NPU4000, paramsConfig,
                         BatchNormLayerTest_NPU4000::getTestCaseName);

}  // namespace
