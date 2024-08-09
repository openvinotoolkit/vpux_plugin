//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

//
// Extra helpers for STL type_traits (partially from C++17 and above)
//

#pragma once

#include <string>
#include <type_traits>

namespace vpux {

//
// Bool logic
//

template <typename T>
using not_ = std::negation<T>;

template <typename... Ts>
using or_ = std::disjunction<Ts...>;

template <typename... Ts>
using and_ = std::conjunction<Ts...>;

//
// enable_if
//

template <typename T, typename... Args>
using enable_t = std::enable_if_t<(Args::value && ...), T>;

template <typename... Args>
using require_t = enable_t<void, Args...>;

template <class T>
struct TypePrinter {
    static constexpr bool hasName() {
        return false;
    }
    static constexpr const char* name();
};

#define TYPE_PRINTER(type)                    \
    template <>                               \
    struct TypePrinter<type> {                \
        static constexpr bool hasName() {     \
            return true;                      \
        }                                     \
        static constexpr const char* name() { \
            return #type;                     \
        }                                     \
    };

TYPE_PRINTER(bool)
TYPE_PRINTER(char)
TYPE_PRINTER(char*)
TYPE_PRINTER(int)
TYPE_PRINTER(unsigned int)
TYPE_PRINTER(int64_t)
TYPE_PRINTER(double)
TYPE_PRINTER(std::string)
}  // namespace vpux
