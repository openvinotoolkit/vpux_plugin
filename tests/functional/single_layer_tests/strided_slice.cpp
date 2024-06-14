//
// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu_ov2_layer_test.hpp>

namespace ov::test {

struct StridedSliceLayerTestParams {
    ov::Shape _in_dims;
    std::vector<int64_t> _begin_data;
    std::vector<int64_t> _end_data;
    std::vector<int64_t> _strides_data;

    std::vector<int64_t> _begin_mask;
    std::vector<int64_t> _end_mask;
    std::vector<int64_t> _new_axis_mask;
    std::vector<int64_t> _shrink_axis_mask;
    std::vector<int64_t> _ellipsis_mask;
};

class StridedSliceLayerTest_FP32_NPU3700 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<StridedSliceLayerTestParams> {
    void SetUp() override {
        const auto test_params = GetParam();
        const ov::Shape inputShape = test_params._in_dims;
        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        std::vector<int64_t> sliceBegin = test_params._begin_data;
        const auto sliceBeginConst =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{sliceBegin.size()}, sliceBegin);

        std::vector<int64_t> sliceEnd = test_params._end_data;
        const auto sliceEndConst = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{sliceEnd.size()}, sliceEnd);

        std::vector<int64_t> sliceStrides = test_params._strides_data;
        const auto sliceStridesConst =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{sliceStrides.size()}, sliceStrides);

        const auto stridedSlice = std::make_shared<ov::op::v1::StridedSlice>(
                params.at(0), sliceBeginConst, sliceEndConst, sliceStridesConst, test_params._begin_mask,
                test_params._end_mask, test_params._new_axis_mask, test_params._shrink_axis_mask,
                test_params._ellipsis_mask);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(stridedSlice)};

        function = std::make_shared<ov::Model>(results, params, "StridedSliceSubGraphTest");
        rel_threshold = 0.5f;
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<StridedSliceLayerTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        return result.str();
    };
};

TEST_P(StridedSliceLayerTest_FP32_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

INSTANTIATE_TEST_SUITE_P(smoke_StridedSlice_fp32, StridedSliceLayerTest_FP32_NPU3700,
                         ::testing::Values(
                                 StridedSliceLayerTestParams{
                                         {1, 3, 64, 64},  // in dims
                                         {0, 0, 0, 0},    // begin data
                                         {1, 3, 64, 64},  // end data
                                         {1, 1, 2, 2},    // strides data
                                         {0, 0, 1, 1},    // begin mask
                                         {1, 0, 1, 1},    // end mask
                                         {0, 0, 0, 0},    // new axis mask
                                         {0, 0, 0, 0},    // shrink axis mask
                                         {0, 0, 0, 0},    // ellipsis mask
                                 },
                                 StridedSliceLayerTestParams{
                                         {1, 3, 64, 64},  // in dims
                                         {0, 0, 0, 16},   // begin data
                                         {1, 3, 64, 32},  // end data
                                         {1, 1, 2, 2},    // strides data
                                         {1, 1, 0, 1},    // begin mask
                                         {1, 1, 0, 1},    // end mask
                                         {0, 0, 0, 0},    // new axis mask
                                         {0, 0, 0, 0},    // shrink axis mask
                                         {0, 0, 0, 0},    // ellipsis mask
                                 }),
                         StridedSliceLayerTest_FP32_NPU3700::getTestCaseName);

}  // namespace ov::test
