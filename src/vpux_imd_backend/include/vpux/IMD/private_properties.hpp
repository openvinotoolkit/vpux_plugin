//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <openvino/runtime/properties.hpp>

namespace ov::intel_npu {

// Path to MV tools
static constexpr ov::Property<std::string, ov::PropertyMutability::RW> mv_tools_path{"VPUX_IMD_MV_TOOLS_PATH"};

// Path to simics directory
static constexpr ov::Property<std::string, ov::PropertyMutability::RW> vpu4_simics_dir{"VPUX_IMD_VPU4_SIMICS_DIR"};

// Launch mode
static constexpr ov::Property<std::string, ov::PropertyMutability::RW> launch_mode{"VPUX_IMD_LAUNCH_MODE"};
static constexpr ov::Property<std::string, ov::PropertyMutability::RW> mv_run_timeout{"VPUX_IMD_MV_RUN_TIMEOUT"};

}  // namespace ov::intel_npu
