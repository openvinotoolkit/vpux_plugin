//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "stdint.h"

namespace vpux::VPU::arch40xx {

constexpr int64_t maxLstmCellHiddenSizeConstant = 128;
constexpr int64_t maxLstmSequenceHiddenSizeConstant = 256;

int64_t getMaxLstmSequenceHiddenSizeConstant() {
    return maxLstmSequenceHiddenSizeConstant;
}

int64_t getMaxLstmCellHiddenSizeConstant() {
    return maxLstmCellHiddenSizeConstant;
}

}  // namespace vpux::VPU::arch40xx
