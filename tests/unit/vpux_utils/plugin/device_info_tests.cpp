//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/plugin/device_info.hpp"

#include <gtest/gtest.h>

using namespace vpux;

TEST(DeviceInfo, CreateFromString) {
    EXPECT_NO_THROW(std::ignore = DeviceInfo("30"));
    EXPECT_NO_THROW(std::ignore = DeviceInfo("30XX"));
    EXPECT_NO_THROW(std::ignore = DeviceInfo("3010"));
    EXPECT_NO_THROW(std::ignore = DeviceInfo("30XXXX"));
    EXPECT_NO_THROW(std::ignore = DeviceInfo("3010A1"));
    EXPECT_NO_THROW(std::ignore = DeviceInfo("3720"));
    EXPECT_NO_THROW(std::ignore = DeviceInfo("NPU3720"));
    EXPECT_NO_THROW(std::ignore = DeviceInfo("npu30xx"));
}

TEST(DeviceInfo, CreateFromString_WrongFormat) {
    EXPECT_ANY_THROW(std::ignore = DeviceInfo(""));
    EXPECT_ANY_THROW(std::ignore = DeviceInfo("QQ"));
    EXPECT_ANY_THROW(std::ignore = DeviceInfo("30AB"));
    EXPECT_ANY_THROW(std::ignore = DeviceInfo("371"));
    EXPECT_ANY_THROW(std::ignore = DeviceInfo("NPU4X1X"));
}

TEST(DeviceInfo, Compare) {
    EXPECT_EQ(DeviceInfo("30"), DeviceInfo::VPUX30XX);
    EXPECT_EQ(DeviceInfo("30xx"), DeviceInfo::VPUX30XX);
    EXPECT_EQ(DeviceInfo("3010A1"), DeviceInfo::VPUX30XX);
    EXPECT_NE(DeviceInfo("npu30xx"), DeviceInfo::VPUX37XX);
}

TEST(DeviceInfo, Print) {
    EXPECT_EQ(printToString("{0}", DeviceInfo::VPUX30XX), "NPU30XX");
}
