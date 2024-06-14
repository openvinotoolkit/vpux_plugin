//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/IMD/private_properties.hpp"

#include "vpux/utils/IE/config.hpp"

#include <chrono>

namespace vpux {

//
// MV_TOOLS_PATH
//

struct MV_TOOLS_PATH final : OptionBase<MV_TOOLS_PATH, std::string> {
    static std::string_view key() {
        return ov::intel_npu::mv_tools_path.name();
    }

    static std::string_view envVar() {
        return "IE_NPU_MV_TOOLS_PATH";
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// VPU4_SIMICS_DIR
//

struct VPU4_SIMICS_DIR final : OptionBase<VPU4_SIMICS_DIR, std::string> {
    static std::string_view key() {
        return ov::intel_npu::vpu4_simics_dir.name();
    }

    static std::string_view envVar() {
        return "VPU4_SIMICS_DIR";
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

//
// LAUNCH_MODE
//

enum class LaunchMode { Simulator, MoviDebug };

std::string_view stringifyEnum(LaunchMode val);

struct LAUNCH_MODE final : OptionBase<LAUNCH_MODE, LaunchMode> {
    static std::string_view key() {
        return ov::intel_npu::launch_mode.name();
    }

    static constexpr std::string_view getTypeName() {
        return "vpux::LaunchMode";
    }

    static std::string_view envVar() {
        return "IE_NPU_IMD_LAUNCH_MODE";
    }

    static LaunchMode parse(std::string_view val);

    static LaunchMode defaultValue() {
        return LaunchMode::Simulator;
    }

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

struct MV_RUN_TIMEOUT final : OptionBase<MV_RUN_TIMEOUT, std::chrono::seconds> {
    static std::string_view key() {
        return ov::intel_npu::mv_run_timeout.name();
    }

    static constexpr std::string_view getTypeName() {
        return "std::chrono::seconds";
    }

    static std::string_view envVar() {
        return "IE_MV_RUN_TIMEOUT";
    }

    static std::chrono::seconds defaultValue();

    static bool isPublic() {
        return false;
    }

    static OptionMode mode() {
        return OptionMode::RunTime;
    }
};

}  // namespace vpux
