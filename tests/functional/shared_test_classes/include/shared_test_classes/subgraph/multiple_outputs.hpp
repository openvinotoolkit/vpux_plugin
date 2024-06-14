//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "vpu_ov2_layer_test.hpp"

using namespace ov::test::utils;

namespace ov::test {

typedef std::tuple<std::vector<size_t>,  // Input Shapes
                   std::vector<size_t>,  // Kernel Shape
                   size_t                // Stride
                   >
        convParams;

typedef std::tuple<ov::element::Type,                   // Network Precision
                   std::string,                         // Target Device
                   std::map<std::string, std::string>,  // Configuration
                   convParams,                          // Convolution Params
                   size_t                               // Output Channels
                   >
        multiOutputTestParams;

class MultioutputTest : public testing::WithParamInterface<multiOutputTestParams>, virtual public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<multiOutputTestParams> obj);
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

protected:
    void SetUp() override;
};

}  // namespace ov::test
