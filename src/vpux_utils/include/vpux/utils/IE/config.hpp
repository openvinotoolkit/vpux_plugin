//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "intel_npu/al/config/config.hpp"
#include "vpux/utils/core/common_logger.hpp"

#include <openvino/runtime/properties.hpp>

namespace vpux {

using intel_npu::Config;
using intel_npu::envVarStrToBool;
using intel_npu::OptionBase;
using intel_npu::OptionMode;
using intel_npu::OptionParser;
using intel_npu::OptionsDesc;

//
// Used to initialize VPUX logger
//

vpux::LogLevel getLogLevel(ov::log::Level);
vpux::LogLevel getLogLevel(const Config&);

}  // namespace vpux
