//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

bool isInputEligibleForConversion(const mlir::Value input, Logger log) {
    if (input.isa<mlir::BlockArgument>()) {
        log.trace("Input is a block argument.");
        return false;
    }
    auto copyOp = input.getDefiningOp<VPUIP::CopyOp>();
    if (copyOp == nullptr) {
        log.trace("Input producer is not a VPUIP.CopyOp.");
        return false;
    }
    if (!VPUIP::isCopyToDDR(copyOp) || !VPUIP::isCopyFromDDR(copyOp)) {
        log.trace("Input producer is not a DDR2DDR copy.");
        return false;
    }
    if (copyOp.getInput().isa<mlir::BlockArgument>()) {
        log.trace("Input copy producer is a block argument.");
        return false;
    }
    auto clusterOp = copyOp.getInput().getDefiningOp<VPUIP::NCEClusterTilingOp>();
    if (clusterOp == nullptr) {
        log.trace("Input copy producer is not a VPUIP.NCEClusterTilingOp.");
        return false;
    }
    auto innerCopy = clusterOp.getInnerTaskOpOfType<VPUIP::CopyOp>();
    if (innerCopy == nullptr) {
        log.trace("VPUIP.NCEClusterTilingOp does not wrap copy.");
        return false;
    }
    auto clusterOpBuffs = clusterOp.getOutputBuffs();
    if (clusterOpBuffs.size() != 1) {
        log.trace("VPUIP.NCEClusterTilingOp is expected to have one and only one output buffer.");
        return false;
    }
    if (clusterOpBuffs[0].isa<mlir::BlockArgument>()) {
        log.trace("VPUIP.NCEClusterTilingOp buffer is a block argument.");
        return false;
    }
    auto clusterOpAlloc = clusterOpBuffs[0].getDefiningOp<mlir::memref::AllocOp>();
    if (clusterOpAlloc == nullptr) {
        log.trace("VPUIP.NCEClusterTilingOp buffer is not allocated via memref.alloc");
        return false;
    }

    log.trace("Input is eligible for conversion");
    return true;
}

VPUIP::NCEClusterTilingOp createTillingCopy(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input,
                                            mlir::Value outputBuff) {
    const auto copyOutBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
    };

    SmallVector<mlir::Value> inputsOutputOperands = {input, outputBuff};
    return rewriter.create<VPUIP::NCEClusterTilingOp>(loc, outputBuff.getType(), inputsOutputOperands,
                                                      copyOutBodyBuilder);
}

//
// FuseCopies
//

class FuseCopies final : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    FuseCopies(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
        setDebugName("FuseCopies");
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// The original subgraph looks like that
// NCE       clusterOpAlloc
//    \      /
//    clusterOp (CMX2DDR)   copyOpAlloc
//        |                /
//        +---------> copyOp (DDR2DDR)   anotherBranch
//                             \            /
//                              origConcatOp
//
// This transformation fuses clusterOp and copyOp into newTilingCopy:
// NCE      copyOpAlloc
//  |      /
// newTilingCopy (CMX2DDR)  anotherBranch
//             \            /
//              origConcatOp
//
mlir::LogicalResult FuseCopies::matchAndRewrite(VPUIP::ConcatViewOp origConcatOp,
                                                mlir::PatternRewriter& rewriter) const {
    const auto concatInputs = origConcatOp.getInputs();
    const auto isEligible = [&](const mlir::Value in) -> bool {
        return isInputEligibleForConversion(in, _log);
    };
    SmallVector<mlir::Value> eligibleInputs;
    std::copy_if(concatInputs.begin(), concatInputs.end(), std::back_inserter(eligibleInputs), isEligible);
    if (eligibleInputs.empty()) {
        return matchFailed(rewriter, origConcatOp, "No DDR2DDR copy to fuse. Skipping ConcatView.");
    }
    rewriter.setInsertionPoint(origConcatOp);
    for (const auto& input : eligibleInputs) {
        auto copyOp = input.getDefiningOp<VPUIP::CopyOp>();
        auto clusterOp = copyOp.getInput().getDefiningOp<VPUIP::NCEClusterTilingOp>();
        auto clusterOpBuffs = clusterOp.getOutputBuffs();
        auto clusterOpAlloc = clusterOpBuffs[0].getDefiningOp<mlir::memref::AllocOp>();
        auto copyOpAlloc = copyOp.getOutputBuff();
        auto newTilingCopy = createTillingCopy(rewriter, copyOp->getLoc(), clusterOp->getOperand(0), copyOpAlloc);
        VPUX_THROW_UNLESS(newTilingCopy.getResults().size() == 1,
                          "VPUIP::NCEClusterTilingOp must have one and only one result");
        copyOp.replaceAllUsesWith(newTilingCopy.getResults()[0]);
        rewriter.eraseOp(copyOp);

        const auto clusterOpResults = clusterOp.getResults();
        const auto hasNoUsers = [](const mlir::Value clusterOpResult) -> bool {
            return clusterOpResult.use_empty();
        };
        if (std::all_of(clusterOpResults.begin(), clusterOpResults.end(), hasNoUsers)) {
            rewriter.eraseOp(clusterOp);
        }
        if (clusterOpAlloc.getMemref().use_empty()) {
            rewriter.eraseOp(clusterOpAlloc);
        }
    }

    return mlir::success();
}

//
// FuseDDRCopiesIntoConcats
//

class FuseDDRCopiesIntoConcats final : public VPUIP::FuseDDRCopiesIntoConcatsBase<FuseDDRCopiesIntoConcats> {
public:
    explicit FuseDDRCopiesIntoConcats(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void FuseDDRCopiesIntoConcats::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseCopies>(&ctx, _log);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createFuseDDRCopiesIntoConcats
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createFuseDDRCopiesIntoConcats(Logger log) {
    return std::make_unique<FuseDDRCopiesIntoConcats>(log);
}
