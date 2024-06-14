//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/IMD/platform_helpers.hpp"

#include "intel_npu/al/config/common.hpp"
#include "vpux/utils/core/error.hpp"

namespace Platform = ov::intel_npu::Platform;

namespace vpux {

namespace {

const std::unordered_map<std::string_view, StringRef> platformToAppNameMap = {
        {Platform::NPU3700, "InferenceManagerDemo_vpu_2_0.elf"},
        {Platform::NPU3720, "InferenceManagerDemo_vpu_2_7.elf"},
        {Platform::NPU4000, "InferenceManagerDemo_vpu_4.elf"},
};

}  // namespace

bool platformSupported(const std::string_view platform) {
    return platformToAppNameMap.find(platform) != platformToAppNameMap.end();
}

StringRef getAppName(const std::string_view platform) {
    const auto it = platformToAppNameMap.find(platform);
    VPUX_THROW_WHEN(it == platformToAppNameMap.end(), "Platform '{0}' is not supported", platform);
    return it->second;
}

}  // namespace vpux
