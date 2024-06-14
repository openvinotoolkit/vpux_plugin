//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
// Class for pretty-logging.
//

#pragma once

#include "vpux/utils/core/common_logger.hpp"
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/helper_macros.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <llvm/Support/WithColor.h>

#include <utility>

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace vpux {

//
// Logging callback
//

using LogCb = FuncRef<void(const formatv_object_base&)>;
void emptyLogCb(const formatv_object_base&);
void globalLogCb(const formatv_object_base&);

//
// Logger
//

class Logger {
public:
    friend struct LoggerAdapter;
    static Logger& global();

public:
    explicit Logger(StringLiteral name, LogLevel lvl);

public:
    Logger nest(size_t inc = 1) const;
    Logger nest(StringLiteral name, size_t inc = 1) const;
    Logger unnest(size_t inc = 1) const;

public:
    auto name() const {
        return _name;
    }

    void setName(StringLiteral name) {
        _name = name;
    }

public:
    auto level() const {
        return _logLevel;
    }

    Logger& setLevel(LogLevel lvl) {
        _logLevel = lvl;
        return *this;
    }

    bool isActive(LogLevel msgLevel) const;

public:
    static llvm::raw_ostream& getBaseStream();
    static llvm::WithColor getLevelStream(LogLevel msgLevel);

public:
    template <typename... Args>
    void fatal(StringLiteral format, Args&&... args) const {
        addEntryPacked(LogLevel::Fatal, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void error(StringLiteral format, Args&&... args) const {
        addEntryPacked(LogLevel::Error, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void warning(StringLiteral format, Args&&... args) const {
        addEntryPacked(LogLevel::Warning, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void info(StringLiteral format, Args&&... args) const {
        addEntryPacked(LogLevel::Info, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void debug(StringLiteral format, Args&&... args) const {
        addEntryPacked(LogLevel::Debug, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void trace(StringLiteral format, Args&&... args) const {
        addEntryPacked(LogLevel::Trace, format, std::forward<Args>(args)...);
    }

public:
    template <typename... Args>
    void addEntry(LogLevel msgLevel, StringLiteral format, Args&&... args) const {
        addEntryPacked(msgLevel, format, std::forward<Args>(args)...);
    }

private:
    void addEntryPackedActive(LogLevel msgLevel, const formatv_object_base& msg) const;

    template <typename... Args>
    void addEntryPacked(LogLevel msgLevel, StringLiteral format, Args&&... args) const {
        if (!isActive(msgLevel)) {
            return;
        }
        addEntryPackedActive(msgLevel, formatv(format.data(), std::forward<Args>(args)...));
    }

private:
    StringLiteral _name;
    LogLevel _logLevel = LogLevel::None;
    size_t _indentLevel = 0;
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    std::string _logFilterStr;
#endif
};

}  // namespace vpux
