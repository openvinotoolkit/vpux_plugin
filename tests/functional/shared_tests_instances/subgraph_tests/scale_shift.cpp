//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/subgraph/scaleshift.hpp>

#include <vector>

#include <common/functions.h>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace {

using namespace ov::test;
using namespace ov::test::utils;

using OVScaleShiftParamsTuple = typename std::tuple<std::vector<ov::Shape>,  // input shapes
                                                    ov::element::Type,       // Model type
                                                    std::string,             // Device name
                                                    std::vector<float>,      // scale
                                                    std::vector<float>>;     // shift

class ScaleShiftSubGraphTestCommon :
        public ::testing::WithParamInterface<OVScaleShiftParamsTuple>,
        virtual public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<OVScaleShiftParamsTuple>& obj) {
        std::vector<ov::Shape> inputShapes;
        ov::element::Type modelType;
        std::vector<float> scale, shift;
        std::tie(inputShapes, modelType, std::ignore, scale, shift) = obj.param;
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;

        result << "IS=" << vec2str(inputShapes) << sep;
        result << "Scale=" << vec2str(scale) << sep;
        result << "Shift=" << vec2str(shift) << sep;
        result << "modelType=" << modelType.get_type_name() << sep;
        return result.str();
    }

protected:
    void SetUp() override {
        std::vector<ov::Shape> inputShapes;
        ov::element::Type type;
        std::vector<float> scale, shift;
        std::tie(inputShapes, type, std::ignore, scale, shift) = this->GetParam();
        auto paramsShape = ov::Shape{1};

        init_input_shapes(static_shapes_to_test_representation({inputShapes}));

        if (inputShapes.size() > 1)
            paramsShape = inputShapes[1];

        ov::ParameterVector paramsIn{std::make_shared<ov::op::v0::Parameter>(type, inputShapes[0])};
        auto mul_const = std::make_shared<ov::op::v0::Constant>(type, paramsShape, scale);
        auto mul = std::make_shared<ov::op::v1::Multiply>(paramsIn[0], mul_const);
        auto add_const = std::make_shared<ov::op::v0::Constant>(type, paramsShape, shift);
        auto add = std::make_shared<ov::op::v1::Add>(mul, add_const);
        function = std::make_shared<ov::Model>(add, paramsIn, "scale_shift");
    }
};

class ScaleShiftSubGraphTest_NPU3700 : public ScaleShiftSubGraphTestCommon {};

TEST_P(ScaleShiftSubGraphTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

}  // namespace

namespace {

std::vector<std::vector<ov::Shape>> inShapes = {
        {{100}},
        {{100}, {100}},
        {{4, 64}, {64}},
        {{1, 8}},
        {{4, 64}},
        {{8, 1024}},
        {{1, 8, 4, 4}, {1, 8, 1, 1}},
        {{1, 128, 32, 32}, {1, 128, 1, 1}},
        {{1, 111, 3, 3}, {1, 111, 1, 1}},
};

std::vector<std::vector<float>> Scales = {{3.0f}, {-3.0f}};

std::vector<std::vector<float>> Shifts = {{3.0f}, {-3.0f}};

std::vector<ov::element::Type> modelTypes = {
        ov::element::f32,
        ov::element::f16,
};

INSTANTIATE_TEST_SUITE_P(smoke_scale_shift_mlir, ScaleShiftSubGraphTest_NPU3700,
                         ::testing::Combine(::testing::ValuesIn(inShapes), ::testing::ValuesIn(modelTypes),
                                            ::testing::Values(DEVICE_NPU), ::testing::ValuesIn(Scales),
                                            ::testing::ValuesIn(Shifts)),
                         ScaleShiftSubGraphTest_NPU3700::getTestCaseName);

}  // namespace
