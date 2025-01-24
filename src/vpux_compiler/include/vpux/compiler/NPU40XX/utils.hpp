//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

namespace vpux {

/// @brief Checks if the ConvertOp is supported on DMA
/// @param convertOp template argument
/// @return boolean

template <typename T>
bool isConvertSupportedOnDMA(T convertOp) {
    auto module = convertOp.getOperation();
    // ConvertSWLayers2VPUIPSWKernelPass still rely on arch check logic here
    // Remove arch check when one-shot enabled, TODO: E#113196
    auto archKind = VPU::getArch(module);
    const std::set<VPU::ArchKind> compatibleTargets = {
            VPU::ArchKind::NPU40XX,
    };
    if (compatibleTargets.count(archKind) <= 0) {
        // Feature is only tested on NPU40XX
        return false;
    }
    auto inputElementType = convertOp.getInput().getType().template cast<NDTypeInterface>().getElementType();
    auto outputElementType = convertOp.getDstElemType();

    return inputElementType.isF32() && (outputElementType.isBF16() || outputElementType.isF16());
}
}  // namespace vpux
