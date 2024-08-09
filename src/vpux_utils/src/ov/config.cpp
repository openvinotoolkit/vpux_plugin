//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/utils/IE/config.hpp"
#include "intel_npu/al/config/common.hpp"

#include <openvino/core/except.hpp>
#include <openvino/runtime/properties.hpp>

namespace vpux {

LogLevel getLogLevel(ov::log::Level level) {
    switch (level) {
    case ov::log::Level::NO:
        return LogLevel::None;
    case ov::log::Level::ERR:
        return LogLevel::Error;
    case ov::log::Level::WARNING:
        return LogLevel::Warning;
    case ov::log::Level::INFO:
        return LogLevel::Info;
    case ov::log::Level::DEBUG:
        return LogLevel::Debug;
    case ov::log::Level::TRACE:
        return LogLevel::Trace;
    }
    // Should not happen unless the enum is extended
    OPENVINO_THROW("Invalid log level.");
}

LogLevel getLogLevel(const intel_npu::Config& config) {
    ov::log::Level ovLevel = config.get<intel_npu::LOG_LEVEL>();
    return getLogLevel(ovLevel);
}

}  // namespace vpux
