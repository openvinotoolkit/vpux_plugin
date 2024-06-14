//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/multiple_outputs.hpp"
#include "common_test_utils/test_constants.hpp"

#include <vector>

#include "vpu_ov2_layer_test.hpp"

using namespace ov::test;
using namespace ov::test::utils;

namespace {
class MultipleoutputTestCommon : public MultioutputTest, virtual public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<multiOutputTestParams>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << MultioutputTest::getTestCaseName(obj) << sep;

        return result.str();
    }
};

class MultipleoutputTest_NPU3700 : public MultipleoutputTestCommon {
    /* tests dumping intermediate outputs

        input
          |
        conv1 -> Output
          |
        conv2
          |
        Pool
          |
        output
    */
};

TEST_P(MultipleoutputTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
};

}  // namespace

namespace {

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const std::vector<std::map<std::string, std::string>> configs = {{{"LOG_LEVEL", "LOG_INFO"}}};

std::vector<convParams> convParam = {
        std::make_tuple(std::vector<size_t>{1, 3, 16, 16},  // InputShape
                        std::vector<size_t>{3, 3},          // KernelShape
                        1)                                  // Stride
};

std::vector<size_t> outputChannels = {16};

INSTANTIATE_TEST_SUITE_P(smoke_MultipleOutputs, MultipleoutputTest_NPU3700,
                         ::testing::Combine(::testing::ValuesIn(modelTypes), ::testing::Values(DEVICE_NPU),
                                            ::testing::ValuesIn(configs), ::testing::ValuesIn(convParam),
                                            ::testing::ValuesIn(outputChannels)),
                         MultipleoutputTestCommon::getTestCaseName);
}  // namespace
