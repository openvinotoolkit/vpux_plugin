// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/single_op/activation.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov {

namespace test {

using namespace ov::test::utils;

// ! [test_convolution:definition]
typedef std::tuple<ov::test::activationParams,
                   std::vector<size_t>,     // Kernel size
                   std::vector<size_t>,     // Strides
                   std::vector<ptrdiff_t>,  // Pad begin
                   std::vector<ptrdiff_t>,  // Pad end
                   std::vector<size_t>,     // Dilation
                   size_t,                  // Num out channels
                   ov::op::PadType>         // Padding type
        convActTestParamsSet;

class ConvActTest : public testing::WithParamInterface<convActTestParamsSet>, virtual public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<convActTestParamsSet>& obj);

protected:
    void SetUp() override;

    void buildFloatFunction();

    void buildFQFunction();
};
// ! [test_convolution:definition]

}  // namespace test
}  // namespace ov
