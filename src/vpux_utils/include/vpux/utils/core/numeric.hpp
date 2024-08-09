//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

//
// Various numeric helper functions.
//

#pragma once

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/type_traits.hpp"

#include <limits>

#include <cassert>
#include <cmath>
#include <cstdint>

namespace vpux {

//
// Integer arithmetic
//

template <typename T, typename = require_t<std::is_integral<T>>>
T isPowerOfTwo(T val) {
    return (val > 0) && ((val & (val - 1)) == 0);
}

template <typename T, typename = require_t<std::is_integral<T>>>
T alignValUp(T val, T align) {
    VPUX_THROW_WHEN(align == 0, "Zero-alignment is not supported");
    const T isPos = static_cast<T>(val >= 0);
    return ((val + isPos * (align - 1)) / align) * align;
}

template <typename T, typename = require_t<std::is_integral<T>>>
T alignValDown(T val, T align) {
    VPUX_THROW_WHEN(align == 0, "Zero-alignment is not supported");
    return (val / align) * align;
}

template <typename T, typename = require_t<std::is_integral<T>>>
T divUp(T a, T b) {
    return (a + b - 1) / b;
}

//
// Float arithmetic
//

inline bool isFloatEqual(float a, float b) {
    return std::fabs(a - b) <= std::numeric_limits<float>::epsilon();
}
inline bool isDoubleEqual(double a, double b) {
    return std::fabs(a - b) <= std::numeric_limits<double>::epsilon();
}

}  // namespace vpux
