//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU30XX/dialect/VPU/impl/nce_sparsity_converters.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"

using namespace vpux;

int32_t VPU::arch30xx::getScale(uint8_t shift, uint16_t mult, double, mlir::Type, VPU::PPETaskAttr ppe) {
    // FIXME: set value when PPE is LPRELU in quant mode
    int32_t PRELU_SCALE_OFFSET = 0;
    int32_t PRELU_SCALE_VALUE = 1;

    int32_t PPE_SHIFT_OFFSET = 8;
    int32_t PPE_SHIFT_VALUE = shift;

    int32_t ROUND_MODE_OFFSET = 14;
    int32_t ROUND_MODE_VALUE = 1;

    int32_t PPE_MULT_OFFSET = 16;
    // FIXME: PPE multiplier has sign, which may affect lower bits
    int32_t PPE_MULT_VALUE = mult;

    int32_t scale = (PRELU_SCALE_VALUE << PRELU_SCALE_OFFSET) | (PPE_SHIFT_VALUE << PPE_SHIFT_OFFSET) |
                    (ROUND_MODE_VALUE << ROUND_MODE_OFFSET) | (PPE_MULT_VALUE << PPE_MULT_OFFSET);

    if (ppe && ppe.getMode().getValue() == VPU::PPEMode::LPRELU) {
        scale &= 0xFFFFFF00;
        scale |= static_cast<int32_t>(ppe.getLreluMult().getInt());
    }

    return scale;
}

int32_t VPU::arch30xx::getBias(double realVal) {
    return VPU::NCESparsity::toFixedPoint(realVal);
}
