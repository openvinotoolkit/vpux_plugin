//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/Types.h>
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

using namespace vpux;

namespace vpux::VPU::arch37xx {

int32_t getScale(uint8_t shift, uint16_t mult, double rescale, mlir::Type inputType);
int32_t getBias(double realVal, mlir::Type inputType);

}  // namespace vpux::VPU::arch37xx
