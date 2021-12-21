// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/prior_box.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"


//tiny misspelling in 'prior_box.hpp'
using namespace LayerTestDefinitions;
//should have been 'LayerTestsDefinitions'

namespace LayerTestsDefinitions {

class KmbPriorBoxLayerTest: public PriorBoxLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
};

TEST_P(KmbPriorBoxLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions


using namespace LayerTestsDefinitions;

namespace {  //openvino eg

const priorBoxSpecificParams param1 = {
   std::vector<float>{16.0},  // min_size
   std::vector<float>{38.46}, // max_size
   std::vector<float>{2.0},   // aspect_ratio
   std::vector<float>{},      // [density]
   std::vector<float>{},      // [fixed_ratio]
   std::vector<float>{},      // [fixed_size]
   false,                     // clip
   true,                      // flip
   16.0,                      // step
   0.5,                       // offset
   std::vector<float>{0.1, 0.1, 0.2, 0.2}, // variance
   false,                     // [scale_all_sizes]
   false                      // min_max_aspect_ratios_order ?
};

INSTANTIATE_TEST_SUITE_P(
        smoke_PriorBox,
        KmbPriorBoxLayerTest,
        testing::Combine(
           testing::Values(param1),
           testing::Values(InferenceEngine::Precision::UNSPECIFIED),
           testing::Values(InferenceEngine::Precision::UNSPECIFIED),
           testing::Values(InferenceEngine::Precision::FP32),
           testing::Values(InferenceEngine::Layout::ANY),
           testing::Values(InferenceEngine::Layout::ANY),
           testing::Values(InferenceEngine::SizeVector{ 24,  42}), //output_size
           testing::Values(InferenceEngine::SizeVector{348, 672}), //image_size
           testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        KmbPriorBoxLayerTest::getTestCaseName
);

}  // namespace
