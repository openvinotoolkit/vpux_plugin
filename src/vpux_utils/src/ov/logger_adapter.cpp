//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <stdarg.h>

#include "vpux/utils/IE/logger_adapter.hpp"
#include "vpux/utils/core/exceptions.hpp"
#include "vpux/utils/core/logger.hpp"

namespace vpux {

LoggerAdapter::LoggerAdapter(std::string_view category) {
    if (category == "Config") {
        impl.reset(new Logger(Logger::global().nest("Config", 0)));
    } else if (category == "OptionsDesc") {
        impl.reset(new Logger(Logger::global().nest("OptionsDesc", 0)));
    } else {
        CORE_VPUX_THROW("This logger instance adapts categoies: \"Config\", \"OptionsDesc\" only"
                        "Please use the traditional \"Logger\" before its refactoring");
    }
}

LoggerAdapter::LoggerAdapter(): impl(new Logger(Logger::global())) {
}

LoggerAdapter::~LoggerAdapter() = default;

void LoggerAdapter::setGlobalLevel(LogLevel lvl) {
    Logger::global().setLevel(lvl);
}

void LoggerAdapter::printImpl(LogLevel level, const std::string& str) {
    impl->addEntryPacked(level, "{0}", str);
}
}  // namespace vpux
