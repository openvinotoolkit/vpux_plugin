//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/dry_run_utils.hpp"
#include "vpux/utils/core/error.hpp"

#include <algorithm>

using namespace vpux;

static constexpr auto MODE_NONE = "none";
static constexpr auto MODE_STUB = "stub";
static constexpr auto MODE_STRIP = "strip";

VPU::DPUDryRunMode VPU::getDPUDryRunMode(std::string mode) {
    std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);

    if (mode == MODE_NONE) {
        return VPU::DPUDryRunMode::NONE;
    } else if (mode == MODE_STUB) {
        return VPU::DPUDryRunMode::STUB;
    } else if (mode == MODE_STRIP) {
        return VPU::DPUDryRunMode::STRIP;
    }

    VPUX_THROW("Unknown value for the DPU dry run option: {0}", mode);
}

VPU::DPUDryRunMode VPU::getDPUDryRunMode(const StrOption& option) {
    auto strOption = convertToOptional(option);
    if (!strOption.has_value()) {
        return VPU::DPUDryRunMode::NONE;
    }
    return getDPUDryRunMode(strOption.value());
}

bool VPU::isDPUDryRunEnabled(const StrOption& option) {
    const auto mode = getDPUDryRunMode(option);
    return mode != VPU::DPUDryRunMode::NONE;
}
