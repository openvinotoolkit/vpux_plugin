//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/ctc_greedy_decoder.hpp"

#include <vector>

#include <common/functions.h>
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov {
namespace test {

class CTCGreedyDecoderLayerTestCommon : public CTCGreedyDecoderLayerTest, virtual public VpuOv2LayerTest {};

class CTCGreedyDecoderLayerTest_NPU3700 : public CTCGreedyDecoderLayerTestCommon {};
class CTCGreedyDecoderLayerTest_NPU3720 : public CTCGreedyDecoderLayerTestCommon {};
class CTCGreedyDecoderLayerTest_NPU4000 : public CTCGreedyDecoderLayerTestCommon {};

void skipInferCallbackImpl(std::stringstream& skip, std::vector<InputShape> inShape) {
    const std::vector<InputShape> badInputShapesForMLIR = {{{}, {{50, 3, 3}}},
                                                           {{}, {{50, 3, 128}}},
                                                           {{}, {{10, 1, 16}}}};
    for (auto iter = badInputShapesForMLIR.cbegin(); iter != badInputShapesForMLIR.cend(); iter++) {
        if (inShape[0].second[0] == iter[0].second[0]) {
            skip << "Comparison fails";
        }
    }
}

TEST_P(CTCGreedyDecoderLayerTest_NPU3700, HW) {
    setSkipInferenceCallback([](std::stringstream& skip) {
        skipInferCallbackImpl(skip, std::get<std::vector<InputShape>>(GetParam()));
    });
    setDefaultHardwareMode();
    run(Platform::NPU3700);
}

TEST_P(CTCGreedyDecoderLayerTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3720);
}

TEST_P(CTCGreedyDecoderLayerTest_NPU4000, SW) {
    setReferenceSoftwareMode();
    run(Platform::NPU4000);
}

}  // namespace test
}  // namespace ov

using namespace ov::test;

namespace {

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,  // Testing FP32/FP16 netPrecision functionality only for small scope of
        ov::element::f16   // tests: GRNLayerTest, SplitLayerTest, CTCGreedyDecoderLayerTest
};

const std::vector<bool> mergeRepeated = {true, false};

// Only batch = 1 is supported
const std::vector<std::vector<ov::Shape>> inputShapes_MLIR = {
        {{88, 1, 71}}, {{10, 1, 16}}, {{50, 3, 3}}, {{50, 3, 128}}, {{1, 1, 16}}};

const auto params_MLIR =
        testing::Combine(testing::ValuesIn(netPrecisions),                                           // Model type
                         testing::ValuesIn(static_shapes_to_test_representation(inputShapes_MLIR)),  // Input shapes
                         testing::ValuesIn(mergeRepeated),                                           // Merge repeated
                         testing::Values(DEVICE_NPU));                                               // Device name

// NPU3700
INSTANTIATE_TEST_SUITE_P(smoke_CTCGreedyDecoder, CTCGreedyDecoderLayerTest_NPU3700, params_MLIR,
                         CTCGreedyDecoderLayerTest::getTestCaseName);

// NPU3720
INSTANTIATE_TEST_SUITE_P(smoke_CTCGreedyDecoder, CTCGreedyDecoderLayerTest_NPU3720, params_MLIR,
                         CTCGreedyDecoderLayerTest::getTestCaseName);

// NPU4000
INSTANTIATE_TEST_SUITE_P(smoke_precommit_CTCGreedyDecoder, CTCGreedyDecoderLayerTest_NPU4000, params_MLIR,
                         CTCGreedyDecoderLayerTest::getTestCaseName);

}  // namespace
