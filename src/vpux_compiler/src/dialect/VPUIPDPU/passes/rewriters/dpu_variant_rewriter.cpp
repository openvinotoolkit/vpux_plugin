//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIPDPU/rewriters/dpu_variant_rewriter.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/dpu_variant_block_rewriters.hpp"

namespace vpux {
namespace VPUIPDPU {

DPUVariantRewriter::DPUVariantRewriter(mlir::MLIRContext* ctx, Logger log, ELF::SymbolReferenceMap& symRefMap)
        : mlir::OpRewritePattern<VPUASM::DPUVariantOp>(ctx), _log(log), _symRefMap(symRefMap) {
    setDebugName("DPUInvariant_VPUIPDPURewriter");
}

mlir::LogicalResult DPUVariantRewriter::matchAndRewrite(VPUASM::DPUVariantOp op,
                                                        mlir::PatternRewriter& rewriter) const {
    auto variant = rewriter.create<VPUIPDPU::DPUVariantOp>(
            op.getLoc(), op.getSymNameAttr(), op.getTaskIndexAttr(), op.getTaskLocationAttr(), op.getNextLinkAttr(),
            op.getInvariantTaskLocationAttr(), op.getWeightsAttr(), op.getWeightTableAttr(), op.getNceTaskTypeAttr(),
            op.getWorkloadIdAttr());

    auto& variantRegion = variant.getRegion();
    rewriter.createBlock(&variantRegion);

    if (DPUVariantIDURewriter(op, rewriter, _log).rewrite(_symRefMap).failed()) {
        return mlir::failure();
    }

    if (DPUVariantODURewriter(op, rewriter, _log).rewrite(_symRefMap).failed()) {
        return mlir::failure();
    }

    auto invariant = mlir::cast<VPUASM::DPUInvariantOp>(_symRefMap.lookupSymbol(op.getInvariant()));
    rewriter.create<VPUIPDPU::BarrierCfgOp>(op.getLoc(), invariant.getWaitBarriers(), invariant.getUpdateBarriers(),
                                            invariant.getStartAfterAttr(), invariant.getCleanAfterAttr());

    auto taskIndex = op.getIndexType().getValue();
    auto isFirstVariant = invariant.getFirstVariantIndex() == taskIndex;
    auto isLastVariant = invariant.getVariantCount() > 1 && invariant.getLastVariantIndex() == taskIndex;
    rewriter.create<VPUIPDPU::DPUGroupOp>(op.getLoc(), invariant.getIndexType(), invariant.getVariantCount(),
                                          isFirstVariant, isLastVariant);

    rewriter.eraseOp(op);

    return mlir::success();
}

}  // namespace VPUIPDPU
}  // namespace vpux
