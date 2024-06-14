//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>

#include <gmock/gmock-matchers.h>

#include "intel_npu/al/config/common.hpp"
#include "intel_npu/al/config/runtime.hpp"
#include "vpux_metrics.hpp"

#include "test_utils/npu_backends_test.hpp"

using MetricsUnitTests = ::testing::Test;
using ::testing::HasSubstr;

TEST_F(MetricsUnitTests, getAvailableDevicesNames) {
    const std::vector<std::string> dummyBackendRegistry = {"npu3720_test_backend"};
    vpux::NPUBackendsTest::Ptr test_backends;
    test_backends = std::make_shared<vpux::NPUBackendsTest>(dummyBackendRegistry);
    std::shared_ptr<vpux::VPUXBackends> backends = std::reinterpret_pointer_cast<vpux::VPUXBackends>(test_backends);
    vpux::Metrics metrics(backends);

    std::vector<std::string> devicesNames = metrics.GetAvailableDevicesNames();

    ASSERT_EQ("3720.dummyDevice", devicesNames[0]);
    ASSERT_EQ("noOtherDevice", devicesNames[1]);
}

TEST_F(MetricsUnitTests, getFullDeviceName) {
    const std::vector<std::string> dummyBackendRegistry = {"npu3720_test_backend"};
    vpux::NPUBackendsTest::Ptr test_backends;
    test_backends = std::make_shared<vpux::NPUBackendsTest>(dummyBackendRegistry);
    std::shared_ptr<vpux::VPUXBackends> backends = std::reinterpret_pointer_cast<vpux::VPUXBackends>(test_backends);
    vpux::Metrics metrics(backends);

    auto device = backends->getDevice();
    EXPECT_THAT(metrics.GetFullDeviceName(device->getName()), HasSubstr("Intel(R) NPU"));
}

TEST_F(MetricsUnitTests, getDeviceUuid) {
    const std::vector<std::string> dummyBackendRegistry = {"npu3720_test_backend"};
    vpux::NPUBackendsTest::Ptr test_backends;
    test_backends = std::make_shared<vpux::NPUBackendsTest>(dummyBackendRegistry);
    std::shared_ptr<vpux::VPUXBackends> backends = std::reinterpret_pointer_cast<vpux::VPUXBackends>(test_backends);
    vpux::Metrics metrics(backends);

    ov::device::UUID testPattern = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x37, 0x20};

    auto device = backends->getDevice();
    ov::device::UUID getDeviceUuid = metrics.GetDeviceUuid(device->getName());

    for (uint64_t i = 0; i < getDeviceUuid.MAX_UUID_SIZE; i++) {
        ASSERT_EQ(testPattern.uuid[i], getDeviceUuid.uuid[i]);
    }
}

TEST_F(MetricsUnitTests, getDeviceArchitecture) {
    const std::vector<std::string> dummyBackendRegistry = {"npu3720_test_backend"};
    vpux::NPUBackendsTest::Ptr test_backends;
    test_backends = std::make_shared<vpux::NPUBackendsTest>(dummyBackendRegistry);
    std::shared_ptr<vpux::VPUXBackends> backends = std::reinterpret_pointer_cast<vpux::VPUXBackends>(test_backends);
    vpux::Metrics metrics(backends);

    auto device = backends->getDevice();
    ASSERT_EQ("3720", metrics.GetDeviceArchitecture(device->getName()));
}

TEST_F(MetricsUnitTests, getBackendName) {
    const std::vector<std::string> dummyBackendRegistry = {"npu3720_test_backend"};
    vpux::NPUBackendsTest::Ptr test_backends;
    test_backends = std::make_shared<vpux::NPUBackendsTest>(dummyBackendRegistry);
    std::shared_ptr<vpux::VPUXBackends> backends = std::reinterpret_pointer_cast<vpux::VPUXBackends>(test_backends);
    vpux::Metrics metrics(backends);

    ASSERT_EQ("NPU3720TestBackend", metrics.GetBackendName());
}
