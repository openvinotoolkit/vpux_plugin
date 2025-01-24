//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/permute_quantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
using namespace vpux;

bool IE::isLegalReorderAddPattern(IE::ReorderOp origOp) {
    if (origOp.getOutput().use_empty()) {
        return false;
    }

    auto opNce = *origOp.getOutput().getUsers().begin();
    // check just 1 child for linked patern
    for (auto user : llvm::make_early_inc_range(origOp.getResult().getUsers())) {
        if (user != opNce) {
            return false;
        }
    }

    if (auto opAdd = mlir::dyn_cast<IE::AddOp>(opNce)) {
        if (opAdd.getInput1() != opAdd.getInput2()) {
            return false;
        }
        if (!opAdd.getOutput().hasOneUse()) {
            return false;
        }
        if (!mlir::isa<IE::QuantizeCastOp>(*opAdd.getOutput().getUsers().begin())) {
            return false;
        }
        return true;
    }

    return false;
}

bool IE::isLegalReorderAvgPoolPattern(IE::ReorderOp origOp) {
    if (!origOp.getOutput().hasOneUse()) {
        return false;
    }
    auto opNce = *origOp.getOutput().getUsers().begin();
    if (auto opPooling = mlir::dyn_cast<IE::AvgPoolOp>(opNce)) {
        return vpux::IE::isQuantizedPurposeAvgPool(opPooling);
    }

    return false;
}

bool IE::isBeneficialConvertToPermuteQuantize(ShapeRef shape) {
    // experiments show that shave is far more performant when C == 1, C == 3 or C == 4 than DMA-MemPermute
    if ((shape[Dims4D::Act::C] != 1) && (shape[Dims4D::Act::C] != 3) && (shape[Dims4D::Act::C] != 4)) {
        return false;
    }
    return shape[Dims4D::Act::N] == 1;
}

std::optional<SmallVector<int64_t>> IE::getAdjustHW(int64_t alignment, int64_t width, int64_t height) {
    if (width > VPU::NCEInvariant::VPU_DIMENSION_LIMIT && height > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        return std::nullopt;
    }
    const auto getHW = [](int64_t lengthOfAlignment, int64_t inputToDivide,
                          int64_t inputToMultiply) -> SmallVector<int64_t> {
        const auto maxFactor = std::max(checked_cast<int64_t>(2), divUp(lengthOfAlignment, checked_cast<int64_t>(2)));
        for (const auto i : irange<int64_t>(2, maxFactor)) {
            if (lengthOfAlignment % i == 0) {
                const auto newShrink = inputToDivide / i;
                if (newShrink > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
                    continue;
                }
                const auto newExpand = inputToMultiply * i;
                if (newExpand > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
                    return {};
                }

                return {newShrink, newExpand};
            }
        }
        return {};
    };

    if (width > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        // For exmaple:
        //     tensor<1x1x1x245760> => tensor<1x1x30x8192>
        //
        const auto numberOfAlignment = width / alignment;
        const auto newHW = getHW(numberOfAlignment, width, height);
        if (newHW.empty()) {
            return std::nullopt;
        }
        return SmallVector<int64_t>{newHW[0], newHW[1]};
    } else if (height > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        // For exmaple:
        //     tensor<1x1x245760x16> => tensor<1x1x8192x480>
        //
        const auto newHW = getHW(height, height, width);
        if (newHW.empty()) {
            return std::nullopt;
        }
        return SmallVector<int64_t>{newHW[1], newHW[0]};
    }

    return std::nullopt;
}

bool IE::isShapeCompatibleWithODUPermute(const ShapeRef shape, const int64_t alignment) {
    if (shape.size() != 4) {
        return false;
    }
    if (shape[Dims4D::Act::N] != 1) {
        return false;
    }
    const auto tensorSizeZ = shape[Dims4D::Act::W];
    return tensorSizeZ % alignment == 0;
}

bool IE::isODUPermuteEffectiveForShape(const ShapeRef shape, const int64_t alignment) {
    // Set alignment to 1 to make alignment check pass all the time.
    // In this case, when isShapeCompatibleWithODUPermute fails, it's not because of the alignment.
    const int64_t neutralAlignment = 1;
    if (!isShapeCompatibleWithODUPermute(shape, neutralAlignment)) {
        return false;
    }

    // E116504: NCEPermute's multi-cluster strategy is manually set as SOK on VPUX40XX which
    // introduces performance issue when dim of H/W is greater than VPU_DIMENSION_LIMIT(8192).
    // Add checking to avoid converting to PermuteQuantize if dims are out of limits unless
    // we could adjust H and W to avoid the limitation
    if (shape[Dims4D::Act::N] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT ||
        shape[Dims4D::Act::C] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        return false;
    }
    // check H or W could adjust or not
    const auto IH = shape[Dims4D::Act::H];
    const auto IW = shape[Dims4D::Act::W];
    if (IH > VPU::NCEInvariant::VPU_DIMENSION_LIMIT || IW > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        auto adjustHW = getAdjustHW(alignment, IW, IH);
        if (!adjustHW.has_value()) {
            return false;
        }
    }

    // Expanding 1xCxHx1 to 1xCxHx16 is not very effective.
    // PermuteQuantize has to process a tensor which is 16 times bigger than the original.
    const int64_t minimalEffectiveWidth = 2;
    return IH * IW % alignment == 0 || IW >= minimalEffectiveWidth;
}
