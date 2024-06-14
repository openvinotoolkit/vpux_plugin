//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/topk.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {

namespace test {

class TopKLayerTestCommon : virtual public TopKLayerTest, virtual public VpuOv2LayerTest {};
class TopKLayerTest_NPU3700 : public TopKLayerTestCommon {};
class TopKLayerTest_NPU3720 : public TopKLayerTestCommon {};
class TopKLayerTest_NPU4000 : public TopKLayerTestCommon {};

TEST_P(TopKLayerTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

TEST_P(TopKLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(TopKLayerTest_NPU4000, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

class TopK1LayerTest_NPU3720 : public TopKLayerTest, virtual public VpuOv2LayerTest {
    void SetUp() override {
        std::vector<InputShape> inputShape;
        ov::element::Type modelType;
        int64_t keepK, axis;
        ov::op::v3::TopK::Mode mode;
        ov::op::v3::TopK::SortType sort;
        std::tie(keepK, axis, mode, sort, modelType, inputShape, targetDevice) = this->GetParam();
        init_input_shapes(inputShape);

        auto param = std::make_shared<ov::op::v0::Parameter>(modelType, inputDynamicShapes.front());
        auto k = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, &keepK);
        auto topk = std::dynamic_pointer_cast<ov::op::v3::TopK>(
                std::make_shared<ov::op::v3::TopK>(param, k, axis, mode, sort));

        ov::ResultVector results;
        for (int i = 0; i < topk->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::op::v0::Result>(topk->output(i)));
        }
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "TopK");
    }
};

TEST_P(TopK1LayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

}  // namespace test

}  // namespace ov

using ov::test::TopK1LayerTest_NPU3720;
using ov::test::TopKLayerTest_NPU3700;
using ov::test::TopKLayerTest_NPU3720;
using ov::test::TopKLayerTest_NPU4000;

namespace {

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const std::vector<int64_t> axes = {0, 1, 2};

const std::vector<int64_t> k = {1, 5, 10};

const std::vector<ov::op::v3::TopK::Mode> modes = {ov::op::v3::TopK::Mode::MIN, ov::op::v3::TopK::Mode::MAX};

const std::vector<ov::op::v3::TopK::SortType> sortTypes = {
        // The implements of SortType::NONE are different.
        // Reference uses std::nth_element and returns k out-of-order values.
        // Kernel returns k data sorted in values. nth_element causes computation increase.
        // ov::op::v3::TopK::SortType::NONE,
        ov::op::v3::TopK::SortType::SORT_INDICES,
        ov::op::v3::TopK::SortType::SORT_VALUES,
};

const auto paramsConfig = ::testing::Combine(
        ::testing::ValuesIn(std::vector<int64_t>{1, 5}), ::testing::ValuesIn(axes), ::testing::ValuesIn(modes),
        ::testing::ValuesIn(sortTypes), ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                ov::test::static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{5, 5, 5}}}))),
        ::testing::Values(ov::test::utils::DEVICE_NPU));

const auto paramsConfigPrecommit = ::testing::Combine(
        ::testing::ValuesIn(std::vector<int64_t>{5}), ::testing::ValuesIn(std::vector<int64_t>{2}),
        ::testing::ValuesIn(modes), ::testing::ValuesIn(sortTypes), ::testing::ValuesIn(modelTypes),
        ::testing::ValuesIn(
                ov::test::static_shapes_to_test_representation(std::vector<std::vector<ov::Shape>>({{{5, 5, 5}}}))),
        ::testing::Values(ov::test::utils::DEVICE_NPU));

// Tracking number [E#85137]
INSTANTIATE_TEST_SUITE_P(smoke_TopK, TopKLayerTest_NPU3700,
                         ::testing::Combine(::testing::ValuesIn(k), ::testing::ValuesIn(axes),
                                            ::testing::ValuesIn(modes), ::testing::ValuesIn(sortTypes),
                                            ::testing::ValuesIn(modelTypes),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                    std::vector<std::vector<ov::Shape>>({{{10, 10, 10}}}))),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU)),
                         TopKLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_TopK, TopKLayerTest_NPU3720, paramsConfig,
                         TopKLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_TopK1, TopK1LayerTest_NPU3720, paramsConfig,
                         TopK1LayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_TopK, TopKLayerTest_NPU4000, paramsConfigPrecommit,
                         TopKLayerTest_NPU4000::getTestCaseName);

// Tiling tests
const std::vector<int64_t> k_Tilling = {1};
const std::vector<int64_t> axes_Tilling = {1};
const std::vector<ov::op::v3::TopK::Mode> modes_Tilling = {ov::op::v3::TopK::Mode::MAX};
const std::vector<ov::op::v3::TopK::SortType> sortTypes_Tilling = {
        ov::op::v3::TopK::SortType::SORT_INDICES,
};
const std::vector<ov::element::Type> modelTypes_Tilling = {ov::element::f16};

INSTANTIATE_TEST_SUITE_P(smoke_TopK_Tilling, TopKLayerTest_NPU3720,
                         ::testing::Combine(::testing::ValuesIn(k_Tilling), ::testing::ValuesIn(axes_Tilling),
                                            ::testing::ValuesIn(modes_Tilling), ::testing::ValuesIn(sortTypes_Tilling),
                                            ::testing::ValuesIn(modelTypes_Tilling),
                                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(
                                                    std::vector<std::vector<ov::Shape>>({{{1, 5, 512, 512}}}))),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU)),
                         TopKLayerTest_NPU3720::getTestCaseName);

}  // namespace
