//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/nce_sparsity_converters.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"

#include <llvm/ADT/bit.h>
#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

int32_t VPU::arch37xx::getScale(uint8_t shift, uint16_t mult, double rescale, mlir::Type inputType, VPU::PPETaskAttr) {
    // VPUX37XX expects scale in IEEE754 format in NCE_DPU_PPE_FP_SCALE register in case input has FP16/BF16 type
    if (inputType.isa<mlir::FloatType>()) {
        return VPU::NCESparsity::toHex(rescale);
    }

    int32_t PPE_SHIFT_OFFSET = 8;
    int32_t PPE_SHIFT_VALUE = shift;

    int32_t PPE_MULT_OFFSET = 16;
    // FIXME: PPE multiplier has sign, which may affect lower bits
    int32_t PPE_MULT_VALUE = mult;

    return (PPE_SHIFT_VALUE << PPE_SHIFT_OFFSET) | (PPE_MULT_VALUE << PPE_MULT_OFFSET);
}

int32_t VPU::arch37xx::getBias(double realVal) {
    return VPU::NCESparsity::toHex(realVal);
}
