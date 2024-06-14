//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <cstddef>

namespace vpux::VPUIP::arch40xx {

// Constants

constexpr size_t DMA_MAX_INDICES_LIST_LENGTH =
        65'536;  // The maximum length of the indices list for scatter-gather addressing on NPU40XX.
constexpr size_t GATHER_DMA_MAX_ELEMENT_SIZE = 4096;
}  // namespace vpux::VPUIP::arch40xx
