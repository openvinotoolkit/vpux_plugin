//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/Types.h>
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/nce_sparsity_converters.hpp"

using namespace vpux;

namespace vpux::VPU::arch37xx {

NCESparsity::IntOrFloatType getScale(uint8_t shift, int16_t mult, double rescale, mlir::Type inputType);
NCESparsity::IntOrFloatType getBias(double realVal, mlir::Type inputType);

}  // namespace vpux::VPU::arch37xx
