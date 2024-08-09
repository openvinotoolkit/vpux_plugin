//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/IMD/device.hpp"
#include "vpux/IMD/executor.hpp"
#include "vpux/IMD/platform_helpers.hpp"
#include "vpux/utils/core/error.hpp"

#include "intel_npu/al/config/common.hpp"

using namespace intel_npu;

namespace intel_npu {

IMDDevice::IMDDevice(const std::string_view platform): _platform(platform) {
    VPUX_THROW_UNLESS(platformSupported(platform), "Unsupported VPUX platform '{0}'", platform);
}

std::shared_ptr<IExecutor> IMDDevice::createExecutor(const std::shared_ptr<const NetworkDescription>& network,
                                                     const Config& config) {
    return std::make_shared<IMDExecutor>(_platform, network, config);
}

std::string IMDDevice::getName() const {
    return ov::intel_npu::Platform::standardize(_platform);
}

std::string IMDDevice::getFullDeviceName() const {
    return "Intel(R) NPU (IMD)";
}

uint32_t IMDDevice::getSubDevId() const {
    return 0;
}

uint32_t IMDDevice::getMaxNumSlices() const {
    if (_platform == ov::intel_npu::Platform::NPU3720) {
        return 2;
    } else if (_platform == ov::intel_npu::Platform::NPU4000) {
        return 6;
    } else {
        VPUX_THROW("Unknown nummber of slices for '{0}'", _platform);
    }
}

}  // namespace intel_npu
