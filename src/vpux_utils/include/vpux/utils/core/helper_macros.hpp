//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

namespace vpux {

//
// VPUX_COMBINE
//

#define VPUX_COMBINE_HELPER2(_X_, _Y_) _X_##_Y_
#define VPUX_COMBINE(_X_, _Y_) VPUX_COMBINE_HELPER2(_X_, _Y_)

#define VPUX_COMBINE_HELPER3(_X_, _Y_, _Z_) _X_##_Y_##_Z_
#define VPUX_COMBINE3(_X, _Y_, _Z_) VPUX_COMBINE_HELPER3(_X_, _Y_, _Z_)

//
// VPUX_UNIQUE_NAME
//

#ifdef __COUNTER__
#define VPUX_UNIQUE_NAME(_BaseName_) VPUX_COMBINE(_BaseName_, __COUNTER__)
#else
#define VPUX_UNIQUE_NAME(_BaseName_) VPUX_COMBINE(_BaseName_, __LINE__)
#endif

//
// VPUX_UNUSED
//

#define VPUX_UNUSED(_var_) (void)(_var_)

//
// VPUX_PACKED
//

#ifdef _MSC_VER
#define VPUX_PACKED(body) __pragma(pack(push, 1)) struct body __pragma(pack(pop))
#elif defined(__GNUC__)
#define VPUX_PACKED(body) struct __attribute__((packed)) body
#endif

//
// VPUX_EXPAND
//

#define VPUX_EXPAND(x) x

}  // namespace vpux
