//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// this header must be used instead direct include of firmware headers

#pragma once

// put everything under generation-specific namespace, as different versions of firmware headers
// define different structures with the same name
// makes it possible to use different firmware headers in the same source file

namespace npu37xx {

// firmware headers are used by projects that cannot include system C headers (e.g. Linux Kernel)
// however, they are using definition from there, such as offsetof and uint32_t
// it may lead to compilation errors, if firmware header is included first (undefined uint32_t)
// to resolve the issue include required system C headers before the firmware ones

#include <cstdint>
#include <cstdlib>

// disable clang-format to keep duplicated headers the same as original source
// simplifies maintenance, such as showing differences between versions
// also helps to deal with compilation issues when clang-format re-arranges include directives,
// so system C header is included after code defined there is used

// clang-format off

// include firmware headers with <>, not "", so on Windows we can ignore warnings, generated from them
#include <details/api/vpu_cmx_info_37xx.h>
#include <details/api/vpu_dma_hw_37xx.h>
#include <details/api/vpu_nce_hw_37xx.h>
#include <details/api/vpu_nnrt_api_37xx.h>

// clang-format on

}  // namespace npu37xx
