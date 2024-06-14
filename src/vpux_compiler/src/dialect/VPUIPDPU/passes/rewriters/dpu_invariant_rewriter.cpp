//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIPDPU/rewriters/dpu_invariant_rewriter.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/dpu_invariant_block_rewriters.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"

namespace vpux {
namespace VPUIPDPU {

DPUInvariantRewriter::DPUInvariantRewriter(mlir::MLIRContext* ctx, Logger log, ELF::SymbolReferenceMap& symRefMap)
        : mlir::OpRewritePattern<VPUASM::DPUInvariantOp>(ctx), _log(log), _symRefMap(symRefMap) {
    setDebugName("DPUInvariant_VPUIPDPURewriter");
}

mlir::LogicalResult DPUInvariantRewriter::matchAndRewrite(VPUASM::DPUInvariantOp op,
                                                          mlir::PatternRewriter& rewriter) const {
    auto inv = rewriter.create<VPUIPDPU::DPUInvariantOp>(
            op.getLoc(), op.getSymNameAttr(), op.getTaskIndexAttr(), op.getTaskLocationAttr(), op.getInputAttr(),
            op.getInputSparsityMapAttr(), op.getInputStorageElementTableAttr(), op.getWeightsAttr(),
            op.getWeightsSparsityMapAttr(), op.getWeightTableAttr(), op.getSprLookupTableAttr(), op.getOutputAttr(),
            op.getOutputSparsityMapAttr(), op.getProfilingDataAttr(), op.getNceTaskTypeAttr(), op.getIsContinuedAttr());

    auto& invRegion = inv.getRegion();
    auto invBlock = rewriter.createBlock(&invRegion);
    std::map<DPUInvariantBlockRewriter::BlockArg, size_t> invBlockArgsPos;

    {
        auto guard = mlir::OpBuilder::InsertionGuard(rewriter);
        if (DPUInvariantBlockRewriter::insertInvBlockArgs(op, invBlock, invBlockArgsPos, _log, _symRefMap).failed()) {
            return mlir::failure();
        }
        if (DPUInvariantIDURewriter(op, invBlock, invBlockArgsPos, rewriter, _log).rewrite().failed()) {
            return mlir::failure();
        }
        if (DPUInvariantMPERewriter(op, invBlock, invBlockArgsPos, rewriter, _log).rewrite().failed()) {
            return mlir::failure();
        }
        if (DPUInvariantPPERewriter(op, invBlock, invBlockArgsPos, rewriter, _log).rewrite().failed()) {
            return mlir::failure();
        }
        if (DPUInvariantODURewriter(op, invBlock, invBlockArgsPos, rewriter, _log).rewrite(_symRefMap).failed()) {
            return mlir::failure();
        }
    }

    rewriter.create<VPUIPDPU::BarrierCfgOp>(op.getLoc(), op.getWaitBarriers(), op.getUpdateBarriers(),
                                            op.getStartAfterAttr(), op.getCleanAfterAttr());

    rewriter.create<VPUIPDPU::DPUGroupOp>(op.getLoc(), op.getIndexType(), op.getVariantCount());

    rewriter.eraseOp(op);

    return mlir::success();
}

}  // namespace VPUIPDPU
}  // namespace vpux
