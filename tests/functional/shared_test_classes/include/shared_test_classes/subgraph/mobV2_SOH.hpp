// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpu_ov2_layer_test.hpp"

namespace ov {

namespace test {

using namespace ov::test::utils;

typedef std::tuple<ov::element::Type,                  // Network Precision
                   std::string,                        // Target Device
                   std::map<std::string, std::string>  // Configuration
                   >
        mobilenetV2SlicedParameters;

class mobilenetV2SlicedTest :
        public testing::WithParamInterface<mobilenetV2SlicedParameters>,
        virtual public VpuOv2LayerTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<mobilenetV2SlicedParameters>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov
