//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/IMD/parsed_properties.hpp"
#include "vpux/utils/core/error.hpp"

namespace intel_npu {

//
// LAUNCH_MODE
//

std::string_view stringifyEnum(LaunchMode val) {
    switch (val) {
    case LaunchMode::Simulator:
        return "Simulator";
    case LaunchMode::MoviDebug:
        return "MoviDebug";
    default:
        return "<UNKNOWN>";
    }
}

LaunchMode LAUNCH_MODE::parse(std::string_view val) {
    if (val == "VPUX_IMD_SIMULATOR") {
        return LaunchMode::Simulator;
    } else if (val == "VPUX_IMD_MOVI_DEBUG") {
        return LaunchMode::MoviDebug;
    }

    VPUX_THROW("Value '{0}' is not a valid VPUX_IMD_LAUNCH_MODE option", val.data());
}

//
// MV_RUN_TIMEOUT
//

std::chrono::seconds MV_RUN_TIMEOUT::defaultValue() {
    using namespace std::chrono_literals;
    static const auto RUN_TIMEOUT = std::chrono::duration_cast<std::chrono::seconds>(60min);
    return RUN_TIMEOUT;
}

}  // namespace intel_npu
