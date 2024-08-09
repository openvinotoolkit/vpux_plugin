//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

// clang-format off

#if defined(_MS_VER)
    #define NPU_PRAGMA(X) __pragma(#X)
#elif defined(__GNUC__) || defined(__clang__)
    #define NPU_PRAGMA(X) _Pragma(#X)
#else
    #define NPU_PRAGMA(X)
#endif

#if defined(_MSC_VER)
    #define NPU_DISABLE_WARNING_PUSH       NPU_PRAGMA(warning(push))
    #define NPU_DISABLE_WARNING_BY(number) NPU_PRAGMA(warning(disable : number))
    #define NPU_DISABLE_WARNING_POP        NPU_PRAGMA(warning(pop))
#elif defined(__GNUC__) || defined(__clang__)
    #define NPU_DISABLE_WARNING_PUSH     NPU_PRAGMA(GCC diagnostic push)
    #define NPU_DISABLE_WARNING_BY(name) NPU_PRAGMA(GCC diagnostic ignored #name)
    #define NPU_DISABLE_WARNING_POP      NPU_PRAGMA(GCC diagnostic pop)
#else
    #define NPU_DISABLE_WARNING_PUSH
    #define NPU_DISABLE_WARNING_BY(X)
    #define NPU_DISABLE_WARNING_POP
#endif

#define NPU_DISABLE_WARNING_START(ticketNumber, id) \
    NPU_DISABLE_WARNING_PUSH                        \
    NPU_DISABLE_WARNING_BY(id)

#define NPU_DISABLE_WARNING_END NPU_DISABLE_WARNING_POP

#if defined(_MSC_VER) || defined(__clang__)
    // -Wmaybe-uninitialized is not supported by the latest clang.
    #define NPU_DISABLE_MAYBE_UNINITIALIZED(ticketNumber) NPU_DISABLE_WARNING_PUSH
#elif defined(__GNUC__)
    #define NPU_DISABLE_MAYBE_UNINITIALIZED(ticketNumber) NPU_DISABLE_WARNING_START(ticketNumber, -Wmaybe-uninitialized)
#else
    #error "Unknown compiler; can't disable warnings"
#endif

#if defined(_MSC_VER) || defined(__clang__)
    // -Wstringop-overflow is not supported by the latest clang.
    #define NPU_DISABLE_STRINGOP_OVERFLOW(ticketNumber) NPU_DISABLE_WARNING_PUSH
#elif defined(__GNUC__)
    #define NPU_DISABLE_STRINGOP_OVERFLOW(ticketNumber) NPU_DISABLE_WARNING_START(ticketNumber, -Wstringop-overflow)
#else
    #error "Unknown compiler; can't disable warnings"
#endif

// clang-format on
