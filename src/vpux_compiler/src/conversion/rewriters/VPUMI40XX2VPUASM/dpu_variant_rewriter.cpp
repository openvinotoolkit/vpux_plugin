//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/dpu_variant_rewriter.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::LogicalResult DPUVariantRewriter::symbolize(VPUMI40XX::DPUVariantOp op, SymbolMapper&,
                                                  mlir::ConversionPatternRewriter& rewriter) const {
    auto symName = findSym(op).getRootReference();
    auto taskLocation = findSym(op.getTaskLocation());
    auto invariantSym = findSym(op.getInvariant());

    auto opUses = op.getResult().getUses();
    mlir::FlatSymbolRefAttr nextLink = nullptr;
    auto nextVariantIt = llvm::find_if(opUses, [](mlir::OpOperand& operand) -> bool {
        auto user = mlir::dyn_cast<VPUMI40XX::DPUVariantOp>(operand.getOwner());
        return user && user.getPreviousTask() == operand.get();
    });
    auto nextVariant =
            nextVariantIt != opUses.end() ? mlir::cast<VPUMI40XX::DPUVariantOp>(nextVariantIt->getOwner()) : nullptr;
    if (nextVariant && nextVariant.isHardLinked()) {
        nextLink = findSym(nextVariant.getTaskLocation());
    }

    auto linkedOp = op.getInvariant().getDefiningOp();
    auto linkedInvariantOp = mlir::dyn_cast<VPUMI40XX::DPUInvariantOp>(linkedOp);
    auto invariantTaskLocation = findSym(linkedInvariantOp.getTaskLocation());

    auto weights = op.getWeights() ? findSym(op.getWeights()) : nullptr;
    auto weightTable = op.getWeightTable() ? findSym(op.getWeightTable()) : nullptr;

    auto taskIdx = mlir::TypeAttr::get(op.getType());

    rewriter.create<VPUASM::DPUVariantOp>(op.getLoc(), symName, taskIdx, taskLocation, nextLink, invariantSym,
                                          invariantTaskLocation, weights, weightTable, op.getNceTaskTypeAttr(),
                                          op.getInStartAttr(), op.getInEndAttr(), op.getStartAttr(), op.getEndAttr(),
                                          op.getPadAttr(), op.getMpeModeAttr(), op.getClusterIdAttr(),
                                          op.getHaloRegionsAttr(), op.getWorkloadIdAttr(), op.getLutReadAttr());

    rewriter.eraseOp(op);

    return mlir::success();
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
