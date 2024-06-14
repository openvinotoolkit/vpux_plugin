//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/mobV2_SOH.hpp"
#include <vector>
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov::test {
class MobilenetV2SlicedSubgraphTestCommon : public mobilenetV2SlicedTest, virtual public VpuOv2LayerTest {
    /* tests for mobilenet v2 split over H unequal subtensors
            input
              |
            groupConv
              |
            Add1
              |
            Clamp
              |
            Conv
              |
            Add2
              |
            output
    */

public:
    static std::string getTestCaseName(const testing::TestParamInfo<mobilenetV2SlicedParameters>& obj) {
        const std::string sep = "_";
        std::ostringstream result;
        result << "TestKind" << ov::test::utils::testKind(__FILE__) << sep;
        result << "TestIdx=" << obj.index << sep;
        result << mobilenetV2SlicedTest::getTestCaseName(obj) << sep;

        return result.str();
    }
};
class MobilenetV2SlicedSubgraphTest_NPU3700 : public MobilenetV2SlicedSubgraphTestCommon {};

TEST_P(MobilenetV2SlicedSubgraphTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(Platform::NPU3700);
};

}  // namespace ov::test

namespace ov::test {

const std::vector<ov::element::Type> modelTypes = {ov::element::f16};

const std::vector<std::map<std::string, std::string>> configs = {{{"LOG_LEVEL", "LOG_INFO"}}};

INSTANTIATE_TEST_CASE_P(smoke_mobilenetV2SlicedTest, MobilenetV2SlicedSubgraphTest_NPU3700,
                        ::testing::Combine(::testing::ValuesIn(modelTypes), ::testing::Values(DEVICE_NPU),
                                           ::testing::ValuesIn(configs)),
                        MobilenetV2SlicedSubgraphTestCommon::getTestCaseName);
}  // namespace ov::test
