//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/utils/core/logger.hpp"

#include "vpux/utils/core/optional.hpp"

#include <llvm/ADT/SmallString.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Regex.h>
#include <llvm/Support/raw_ostream.h>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <optional>

using namespace vpux;

//
// LogCb
//

void vpux::emptyLogCb(const formatv_object_base&) {
}

void vpux::globalLogCb(const formatv_object_base& msg) {
    Logger::global().trace("{0}", msg.str());
}

//
// Logger
//
static const char* logLevelPrintout[] = {"NONE", "FATAL", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE"};

Logger& vpux::Logger::global() {
    static std::optional<Logger> globalLogger = std::nullopt;

    if (globalLogger.has_value()) {
        return globalLogger.value();
    }

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    LogLevel logLvl = LogLevel::Warning;
    if (const auto env = std::getenv("OV_NPU_LOG_LEVEL")) {
        auto logStr = std::string(env);
        if (logStr == "LOG_NONE") {
            logLvl = LogLevel::None;
        } else if (logStr == "LOG_ERROR") {
            logLvl = LogLevel::Error;
        } else if (logStr == "LOG_WARNING") {
            logLvl = LogLevel::Warning;
        } else if (logStr == "LOG_INFO") {
            logLvl = LogLevel::Info;
        } else if (logStr == "LOG_DEBUG") {
            logLvl = LogLevel::Debug;
        } else if (logStr == "LOG_TRACE") {
            logLvl = LogLevel::Trace;
        }
    }
    globalLogger = Logger("global", logLvl);
#else
    globalLogger = Logger("global", LogLevel::None);
#endif

    return globalLogger.value();
}

vpux::Logger::Logger(StringLiteral name, LogLevel lvl): _name(name), _logLevel(lvl) {
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    if (const auto env = std::getenv("IE_NPU_LOG_FILTER")) {
        _logFilterStr = std::string(env);
    }
#endif
}

Logger vpux::Logger::nest(size_t inc) const {
    return nest(name(), inc);
}

Logger vpux::Logger::nest(StringLiteral name, size_t inc) const {
    Logger nested(name, level());
    nested._indentLevel = _indentLevel + inc;
    return nested;
}

Logger vpux::Logger::unnest(size_t inc) const {
    assert(_indentLevel >= inc);
    Logger unnested(name(), level());
    unnested._indentLevel = _indentLevel - inc;
    return unnested;
}

bool vpux::Logger::isActive(LogLevel msgLevel) const {
#if !defined(NDEBUG)
    if (llvm::DebugFlag && llvm::isCurrentDebugType(name().data())) {
        return true;
    }
#endif

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    if (static_cast<int32_t>(msgLevel) > static_cast<int32_t>(_logLevel)) {
        return false;
    }

    static const auto logFilter = [&]() -> llvm::Regex {
        if (!_logFilterStr.empty()) {
            const StringRef filter(_logFilterStr);

            if (!filter.empty()) {
                return llvm::Regex(filter, llvm::Regex::IgnoreCase);
            }
        }
        return {};
    }();

    if (logFilter.isValid()) {
        return logFilter.match(_name);
    }

    return true;
#else
    return static_cast<int32_t>(msgLevel) <= static_cast<int32_t>(_logLevel);
#endif
}

llvm::raw_ostream& vpux::Logger::getBaseStream() {
#ifdef NDEBUG
    return llvm::outs();
#else
    return llvm::DebugFlag ? llvm::dbgs() : llvm::outs();
#endif
}

namespace {

llvm::raw_ostream::Colors getColor(LogLevel msgLevel) {
    switch (msgLevel) {
    case LogLevel::Fatal:
    case LogLevel::Error:
        return llvm::raw_ostream::RED;
    case LogLevel::Warning:
        return llvm::raw_ostream::YELLOW;
    case LogLevel::Info:
        return llvm::raw_ostream::CYAN;
    case LogLevel::Debug:
    case LogLevel::Trace:
        return llvm::raw_ostream::GREEN;
    default:
        return llvm::raw_ostream::SAVEDCOLOR;
    }
}

}  // namespace

llvm::WithColor vpux::Logger::getLevelStream(LogLevel msgLevel) {
    const auto color = getColor(msgLevel);
    return llvm::WithColor(getBaseStream(), color, true, false, llvm::ColorMode::Auto);
}

void vpux::Logger::addEntryPackedActive(LogLevel msgLevel, const formatv_object_base& msg) const {
    llvm::SmallString<512> tempBuf;
    llvm::raw_svector_ostream tempStream(tempBuf);

    char timeStr[] = "undefined_time";
    time_t now = time(nullptr);
    struct tm* loctime = localtime(&now);
    if (loctime != nullptr) {
        strftime(timeStr, sizeof(timeStr), "%H:%M:%S", loctime);
    }

    using namespace std::chrono;
    uint32_t ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() % 1000;

    printTo(tempStream, "[{0}] {1}.{2,0+3} [{3}] ", logLevelPrintout[static_cast<uint8_t>(msgLevel)], timeStr, ms,
            _name);

    for (size_t i = 0; i < _indentLevel; ++i)
        tempStream << "  ";

    msg.format(tempStream);
    tempStream << "\n";

    static std::mutex logMtx;
    std::lock_guard<std::mutex> logMtxLock(logMtx);

    auto colorStream = getLevelStream(msgLevel);
    auto& stream = colorStream.get();
    stream << tempStream.str();
    stream.flush();
}
