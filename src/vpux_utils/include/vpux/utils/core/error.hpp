//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

//
// Error handling - helper macros to throw exceptions.
//

#pragma once

#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/helper_macros.hpp"

#include <llvm/Support/Compiler.h>

namespace vpux {

//
// Exceptions
//

class Exception : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

class WlmRollbackException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

namespace details {
template <typename ExceptionT>
[[noreturn]] void throwFormat(const char* file, int line, const std::string& message);

}  // namespace details

#define VPUX_THROW_TYPED(_exception_, ...) \
    vpux::details::throwFormat<_exception_>(__FILE__, __LINE__, vpux::printToString(__VA_ARGS__))

#define VPUX_THROW_TYPED_UNLESS(_exception_, _condition_, ...) \
    if (LLVM_UNLIKELY(!(_condition_)))                         \
    VPUX_THROW_TYPED(_exception_, __VA_ARGS__)

#define VPUX_THROW_TYPED_WHEN(_exception_, _condition_, ...) \
    if (LLVM_UNLIKELY(_condition_))                          \
    VPUX_THROW_TYPED(_exception_, __VA_ARGS__)

#define VPUX_THROW(...) VPUX_THROW_TYPED(vpux::Exception, __VA_ARGS__)

#define VPUX_THROW_UNLESS(_condition_, ...) VPUX_THROW_TYPED_UNLESS(vpux::Exception, _condition_, __VA_ARGS__)

#define VPUX_THROW_WHEN(_condition_, ...) VPUX_THROW_TYPED_WHEN(vpux::Exception, _condition_, __VA_ARGS__)

}  // namespace vpux
