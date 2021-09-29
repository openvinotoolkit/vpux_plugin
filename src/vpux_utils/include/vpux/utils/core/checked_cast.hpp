//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

//
// Safe version of `static_cast` with run-time checks.
//

#pragma once

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/type_traits.hpp"

#include <llvm/Support/TypeName.h>

#include <limits>

namespace vpux {

namespace details {

template <bool Cond, class Func>
enable_if_t<Cond> staticIf(Func&& func) {
    func();
}

template <bool Cond, class Func>
enable_if_t<!Cond> staticIf(Func&&) {
}

// To overcame syntax parse error, when `>` comparison operator is threated as
// template closing bracket
template <typename T1, typename T2>
constexpr bool Greater(T1&& v1, T2&& v2) {
    return v1 > v2;
}

}  // namespace details

template <typename OutT, typename InT>
enable_t<OutT, std::is_same<OutT, InT>> checked_cast(InT value) {
    return value;
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_integral<InT>, std::is_signed<InT>, std::is_integral<OutT>, std::is_signed<OutT>,
         not_<std::is_same<OutT, InT>>>
checked_cast(InT value) {
    details::staticIf<std::numeric_limits<InT>::lowest() < std::numeric_limits<OutT>::lowest()>([&] {
        VPUX_THROW_UNLESS(value >= std::numeric_limits<OutT>::lowest(), "Can not safely cast {0} from {1} to {2}",
                          static_cast<int64_t>(value), llvm::getTypeName<InT>(), llvm::getTypeName<OutT>());
    });

    details::staticIf<details::Greater(std::numeric_limits<InT>::max(), std::numeric_limits<OutT>::max())>([&] {
        VPUX_THROW_UNLESS(value <= std::numeric_limits<OutT>::max(), "Can not safely cast {0} from {1} to {2}",
                          static_cast<int64_t>(value), llvm::getTypeName<InT>(), llvm::getTypeName<OutT>());
    });

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_integral<InT>, std::is_unsigned<InT>, std::is_integral<OutT>, std::is_unsigned<OutT>,
         not_<std::is_same<OutT, InT>>>
checked_cast(InT value) {
    details::staticIf<details::Greater(std::numeric_limits<InT>::max(), std::numeric_limits<OutT>::max())>([&] {
        VPUX_THROW_UNLESS(value <= std::numeric_limits<OutT>::max(), "Can not safely cast {0} from {1} to {2}",
                          static_cast<uint64_t>(value), llvm::getTypeName<InT>(), llvm::getTypeName<OutT>());
    });

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_integral<InT>, std::is_unsigned<InT>, std::is_integral<OutT>, std::is_signed<OutT>> checked_cast(
        InT value) {
    details::staticIf<details::Greater(std::numeric_limits<InT>::max(),
                                       static_cast<std::make_unsigned_t<OutT>>(std::numeric_limits<OutT>::max()))>([&] {
        VPUX_THROW_UNLESS(value <= static_cast<std::make_unsigned_t<OutT>>(std::numeric_limits<OutT>::max()),
                          "Can not safely cast {0} from {1} to {2}", static_cast<uint64_t>(value),
                          llvm::getTypeName<InT>(), llvm::getTypeName<OutT>());
    });

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_integral<InT>, std::is_signed<InT>, std::is_integral<OutT>, std::is_unsigned<OutT>> checked_cast(
        InT value) {
    if (value <0) {
        printf("<0\n");
    }
    VPUX_THROW_UNLESS(value >= 0, "Can not safely cast {0} from {1} to {2}", static_cast<int64_t>(value),
                      llvm::getTypeName<InT>(), llvm::getTypeName<OutT>());

    details::staticIf<details::Greater(static_cast<std::make_unsigned_t<InT>>(std::numeric_limits<InT>::max()),
                                       std::numeric_limits<OutT>::max())>([&] {
        VPUX_THROW_UNLESS(static_cast<std::make_unsigned_t<InT>>(value) <= std::numeric_limits<OutT>::max(),
                          "Can not safely cast {0} from {1} to {2}", static_cast<int64_t>(value),
                          llvm::getTypeName<InT>(), llvm::getTypeName<OutT>());
    });

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_floating_point<InT>, std::is_integral<OutT>> checked_cast(InT value) {
    VPUX_THROW_UNLESS(value <= static_cast<InT>(std::numeric_limits<OutT>::max()),
                      "Can not safely cast {0} from {1} to {2}", value, llvm::getTypeName<InT>(),
                      llvm::getTypeName<OutT>());

    VPUX_THROW_UNLESS(value >= static_cast<InT>(std::numeric_limits<OutT>::lowest()),
                      "Can not safely cast {0} from {1} to {2}", value, llvm::getTypeName<InT>(),
                      llvm::getTypeName<OutT>());

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_integral<InT>, std::is_signed<InT>, std::is_floating_point<OutT>> checked_cast(InT value) {
    VPUX_THROW_UNLESS(static_cast<InT>(static_cast<OutT>(value)) == value, "Can not safely cast {0} from {1} to {2}",
                      static_cast<int64_t>(value), llvm::getTypeName<InT>(), llvm::getTypeName<OutT>());

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_integral<InT>, std::is_unsigned<InT>, std::is_floating_point<OutT>> checked_cast(InT value) {
    VPUX_THROW_UNLESS(static_cast<InT>(static_cast<OutT>(value)) == value, "Can not safely cast {0} from {1} to {2}",
                      static_cast<uint64_t>(value), llvm::getTypeName<InT>(), llvm::getTypeName<OutT>());

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_same<double, InT>, std::is_same<float, OutT>> checked_cast(InT value) {
    VPUX_THROW_UNLESS(static_cast<InT>(static_cast<OutT>(value)) == value, "Can not safely cast {0} from {1} to {2}",
                      value, llvm::getTypeName<InT>(), llvm::getTypeName<OutT>());

    return static_cast<OutT>(value);
}

template <typename OutT, typename InT>
enable_t<OutT, std::is_same<float, InT>, std::is_same<double, OutT>> checked_cast(InT value) {
    return static_cast<OutT>(value);
}

}  // namespace vpux
