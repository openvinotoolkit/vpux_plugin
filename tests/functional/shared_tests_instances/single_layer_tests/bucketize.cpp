//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "single_op_tests/bucketize.hpp"
#include <vector>
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class BucketizeLayerTestCommon : public BucketizeLayerTest, virtual public VpuOv2LayerTest {
    void SetUp() override {
        std::tie(std::ignore /*Data shape*/, std::ignore /*Right edge of interval*/,
                 std::ignore /*Data input precision*/, std::ignore /*Buckets input precision*/, outType,
                 std::ignore /*deviceName*/
                 ) = GetParam();

        BucketizeLayerTest::SetUp();
    }
};

class BucketizeLayerTest_NPU3720 : public BucketizeLayerTestCommon {};
class BucketizeLayerTest_NPU4000 : public BucketizeLayerTestCommon {};

TEST_P(BucketizeLayerTest_NPU3720, SW) {
    setSkipCompilationCallback([](std::stringstream& skip) {
        const auto outputType = std::get<4>(GetParam());
        if (outputType == ov::element::i64) {
            skip << "I64 Precision is not supported yet!";
        }
    });
    setReferenceSoftwareMode();
    run(Platform::NPU3720);
}

TEST_P(BucketizeLayerTest_NPU4000, SW) {
    setSkipCompilationCallback([](std::stringstream& skip) {
        const auto outputType = std::get<4>(GetParam());
        if (outputType == ov::element::i64) {
            skip << "I64 Precision is not supported yet!";
        }
    });
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> dataInputPrecisions = {
        ov::element::f16,
        ov::element::f32,
        ov::element::i32,
};

const std::vector<ov::element::Type> bucketsInputPrecisions = {
        ov::element::f16,
        ov::element::f32,
        ov::element::i32,
};

const std::vector<ov::element::Type> outputPrecisions = {
        ov::element::i32,
        ov::element::i64,  // Skipped before load
};

const std::vector<std::vector<ov::Shape>> inputShapes = {{{1, 20, 20}, {100}}, {{2, 3, 50, 50}, {100}}};

const std::vector<bool> with_right_bound = {true, false};

const auto testBucketizeParamsI64 =
        ::testing::Combine(::testing::Values(ov::test::static_shapes_to_test_representation(inputShapes[0])),
                           ::testing::ValuesIn(with_right_bound), ::testing::Values(dataInputPrecisions[0]),
                           ::testing::Values(bucketsInputPrecisions[0]), ::testing::Values(outputPrecisions[1]),
                           ::testing::Values(DEVICE_NPU));

const auto testBucketizeParams =
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes)),
                           ::testing::ValuesIn(with_right_bound), ::testing::Values(dataInputPrecisions[0]),
                           ::testing::Values(bucketsInputPrecisions[0]), ::testing::Values(outputPrecisions[0]),
                           ::testing::Values(DEVICE_NPU));

// 3720
INSTANTIATE_TEST_CASE_P(smoke_precomit_BucketizeTest, BucketizeLayerTest_NPU3720, testBucketizeParams,
                        BucketizeLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_BucketizeTestI64, BucketizeLayerTest_NPU3720, testBucketizeParamsI64,
                        BucketizeLayerTest_NPU3720::getTestCaseName);

// 4000
INSTANTIATE_TEST_CASE_P(smoke_precommit_BucketizeTest, BucketizeLayerTest_NPU4000, testBucketizeParams,
                        BucketizeLayerTest_NPU4000::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_BucketizeTestI64, BucketizeLayerTest_NPU4000, testBucketizeParamsI64,
                        BucketizeLayerTest_NPU4000::getTestCaseName);

}  // namespace
