//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;
namespace {

//
// ParallelCopiesRewriter
//

class ParallelCopiesRewriter final : public mlir::OpRewritePattern<VPUIP::CopyOp> {
public:
    ParallelCopiesRewriter(mlir::MLIRContext* ctx, Logger log, const bool enableOptimizeConstCopy)
            : mlir::OpRewritePattern<VPUIP::CopyOp>(ctx), _log(log), _enableOptimizeConstCopy(enableOptimizeConstCopy) {
        setDebugName("ParallelCopiesRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isLegalAndBenifitParallelCopiesRewriter(VPUIP::CopyOp origOp, NDTypeInterface inputType,
                                                 NDTypeInterface outputType, VPU::NCEInterpolateModeAttr modeAttr,
                                                 IE::InterpolateCoordModeAttr coordModeAttr) const;

    Logger _log;
    bool _enableOptimizeConstCopy;
};

// Utility function to extract VPUIP::CopyOp from NCEClusterTilingOp. If op isnt NCEClusterTilingOp or didnt contain
// CopyOp return nullptr
mlir::Operation* getCopyOpFromClusterTilingOp(mlir::Operation* op) {
    if (auto nceClusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(op)) {
        return mlir::dyn_cast<VPUIP::CopyOp>(nceClusterTilingOp.getInnerTaskOp());
    }
    return nullptr;
}

mlir::Operation* getParentOfCopyOp(VPUIP::CopyOp copyOp) {
    // Check is CopyOp wrapper by NCECluster task
    if (auto wrapperOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(copyOp->getParentOp())) {
        return wrapperOp.getInputs()[0].getDefiningOp();
    }
    return copyOp.getInput().getDefiningOp();
}

// Utility function to get DistributedBufferType from NCEClusterTilingOp. If op isnt NCEClusterTilingOp or return type
// isnt DistributedBufferType return empty value
std::optional<VPUIP::DistributedBufferType> getClusterCopyResultType(mlir::Operation* op) {
    if (auto nceClusterTilingOp = mlir::dyn_cast_or_null<VPUIP::NCEClusterTilingOp>(op)) {
        const auto resType = VPUIP::extractDataType(nceClusterTilingOp->getResult(0));
        if (resType.isa<VPUIP::DistributedBufferType>()) {
            return resType.cast<VPUIP::DistributedBufferType>();
        }
    }
    return std::nullopt;
}

bool hasSiblingCopyFusable(VPUIP::SubViewOp subViewOp, VPUIP::CopyOp copyOp, mlir::Operation* parentOp, Logger log) {
    bool hasSiblingCopy = false;
    if (parentOp == nullptr || parentOp->getNumResults() <= 0) {
        log.trace("Is not fusable because haven't consumers or parent is empty");
        return false;
    }
    for (auto siblingOp : parentOp->getResult(0).getUsers()) {
        if (auto wrappedCopyOp = getCopyOpFromClusterTilingOp(siblingOp)) {
            siblingOp = wrappedCopyOp;
        } else {
            if (!mlir::isa<VPUIP::CopyOp>(*siblingOp)) {
                if (!mlir::isa<VPUIP::SubViewOp>(*siblingOp)) {
                    continue;
                } else {
                    // TODO: E#116963
                    auto childOfSiblingOp = to_vector(siblingOp->getResult(0).getUsers()).back();
                    if (auto wrappedCopyOp = getCopyOpFromClusterTilingOp(childOfSiblingOp)) {
                        childOfSiblingOp = wrappedCopyOp;
                    }
                    if (!mlir::isa<VPUIP::CopyOp>(childOfSiblingOp)) {
                        continue;
                    }
                    // match SubView->Copy
                    if (subViewOp == nullptr) {
                        continue;
                    }
                    auto siblingSubViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(siblingOp);
                    if (parseIntArrayAttr<int64_t>(subViewOp.getStaticOffsets()) !=
                                parseIntArrayAttr<int64_t>(siblingSubViewOp.getStaticOffsets()) ||
                        parseIntArrayAttr<int64_t>(subViewOp.getStaticSizes()) !=
                                parseIntArrayAttr<int64_t>(siblingSubViewOp.getStaticSizes())) {
                        continue;
                    }
                    siblingOp = childOfSiblingOp;
                }
            }
        }

        // Check 3: current op's consumers are copied to DDR immediately after execution
        for (const auto childOfSiblingOp : siblingOp->getResult(0).getUsers()) {
            if (childOfSiblingOp->use_empty()) {
                log.trace("Is not fusable because childOfSiblingOp haven't consumers");
                return false;
            }
            for (const auto grandChildOfSiblingOp : childOfSiblingOp->getResult(0).getUsers()) {
                auto concatOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(grandChildOfSiblingOp);
                // If the ChildOfSiblingOp is a multi-shaveOp there will be a ConcatViewOp after ChildOfSiblingOp, skip
                // this ConcatViewOp and continue the optimization.
                auto childCopyOfSiblingOp =
                        (concatOp != nullptr) ? mlir::dyn_cast<VPUIP::CopyOp>(*(concatOp.getOutput().user_begin()))
                                              : mlir::dyn_cast<VPUIP::CopyOp>(grandChildOfSiblingOp);
                if (childCopyOfSiblingOp == nullptr) {
                    log.trace("Is not fusable because childOfSiblingOp is not CopyOp");
                    return false;
                }
                const auto input = childCopyOfSiblingOp.getInput().getType().cast<vpux::NDTypeInterface>();
                const auto output = childCopyOfSiblingOp.getOutput().getType().cast<vpux::NDTypeInterface>();
                if (input.getMemoryKind() != VPU::MemoryKind::CMX_NN ||
                    output.getMemoryKind() != VPU::MemoryKind::DDR) {
                    log.trace("Is not fusable because childCopyOfSiblingOp is not CMX->DDR copy");
                    return false;
                }
            }
        }

        if (siblingOp != copyOp) {
            hasSiblingCopy = true;
        }
    }
    return hasSiblingCopy;
}

bool isCopyFusable(VPUIP::CopyOp copyOp, bool enableOptimizeConstCopy, Logger log) {
    // Check 1: copy DDR->CMX
    const auto srcMemory = copyOp.getInput().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
    const auto dstMemory = copyOp.getOutput().getType().cast<vpux::NDTypeInterface>().getMemoryKind();
    if (srcMemory == dstMemory || srcMemory == VPU::MemoryKind::CMX_NN) {
        log.trace("Is not fusable because not DDR->CMX copy");
        return false;
    }

    // Check 2: parallel
    // All the consumers of the parent op should be copies
    // At least one more copy except for the current one
    auto parentOp = getParentOfCopyOp(copyOp);
    if (parentOp == nullptr) {
        log.trace("Is not fusable because haven't parentOp");
        return false;
    }

    auto wrapperOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(copyOp->getParentOp());
    auto copyUsers = wrapperOp != nullptr ? wrapperOp->getUsers() : copyOp->getUsers();
    for (auto* user : copyUsers) {
        while (VPUIP::isPureViewOp(user)) {
            if (mlir::isa<VPUIP::ConcatViewOp>(user)) {
                // If usage is through concat operation then optimization cannot be performed because
                // concat with different inputs requires different output buffers and each needs to be handled
                // by dedicated copy, which will refer to different output buffer
                log.trace("Is not fusable because user is concat op");
                return false;
            } else {
                if (user->getUsers().empty()) {
                    break;
                }
                user = *user->getUsers().begin();
            }
        }
    }

    // Optimize copies for weights. If serveral convolutions share same weights, the weight copies can be optimized with
    // single copy e.g. cases when the NCEOps that shares the same weights
    // Note that weight table and compressed convolution cannot apply this optimization. This is because
    // 1. for weight table, contents of weigthTable need to be adjusted with proper pointer value
    // 2. for compressed convolution, const data like weight also will be adjusted in ConvWeightsCompression pass,
    // will prevent the copy optimization.
    if (mlir::isa<Const::DeclareOp>(parentOp)) {
        if (!enableOptimizeConstCopy) {
            log.trace("Is not fusable because enableOptimizeConstCopy is not enabled");
            return false;
        }
        auto copyOutput = wrapperOp != nullptr ? wrapperOp->getResult(0) : copyOp.getOutput();
        for (const auto& user : copyUsers) {
            if (auto nceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(user)) {
                if (nceOp.getWeights() != copyOutput || VPUIP::canWeightsBeCompressed(nceOp) ||
                    nceOp.getWeightsSparsityMap() != nullptr) {
                    log.trace("Is not fusable because copyOutput is not weights or weights can be compressed");
                    return false;
                }
            } else if (auto tiledNceOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(user)) {
                // check for NCE multicluster task, no need for Shave multicluster task
                auto innerNceOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(tiledNceOp.getInnerTaskOp());
                if (innerNceOp != nullptr) {
                    auto weights = VPUIP::getTopBufferOfNCEClusterTiling(innerNceOp, innerNceOp.getWeights());
                    if (copyOutput != weights) {
                        log.trace("Is not fusable because copyOutput is not weights");
                        return false;
                    }
                    if (VPUIP::canTilingWeightsBeCompressed(tiledNceOp)) {
                        log.trace("Is not fusable because tiling weights can be compressed");
                        return false;
                    }
                }
            }
        }
    }

    auto subViewFusable = false;
    if (auto subViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(parentOp)) {
        subViewFusable = hasSiblingCopyFusable(subViewOp, copyOp, subViewOp.getSource().getDefiningOp(), log);
    }
    // We have 2 calls here, one to check if we have SubViewOp 1..n SubviewOp
    // Other for TilingCopy 1..n TilingCopy
    if (!(subViewFusable || hasSiblingCopyFusable(nullptr, copyOp, parentOp, log))) {
        log.trace("Is not fusable because doesn't have fusable sibling");
        return false;
    }

    return true;
}

mlir::LogicalResult ParallelCopiesRewriter::matchAndRewrite(VPUIP::CopyOp copyOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), copyOp->getName(), copyOp->getLoc());

    auto nestedLogger = _log.nest();
    if (!isCopyFusable(copyOp, _enableOptimizeConstCopy, nestedLogger)) {
        return mlir::failure();
    }

    auto copyReplaceTarget = copyOp.getOperation();
    bool isClusterCopy = false;
    auto srcClusterCopyOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(copyOp->getParentOp());
    if (srcClusterCopyOp != nullptr) {
        isClusterCopy = true;
        copyReplaceTarget = srcClusterCopyOp.getOperation();
    }

    const auto isSameSubViewFunc = [](VPUIP::SubViewOp srcSubView, VPUIP::SubViewOp siblingSubView) {
        if (srcSubView == siblingSubView) {
            return false;
        }

        return (srcSubView.getStaticOffsets() == siblingSubView.getStaticOffsets()) &&
               (srcSubView.getStaticSizes() == siblingSubView.getStaticSizes()) &&
               (srcSubView.getStaticStrides() == siblingSubView.getStaticStrides());
    };

    const auto isSameCopyFunc = [&](VPUIP::CopyOp srcCopyOp, mlir::Operation* op) {
        auto siblingCopy = isClusterCopy ? mlir::dyn_cast_or_null<VPUIP::CopyOp>(getCopyOpFromClusterTilingOp(op))
                                         : mlir::dyn_cast<VPUIP::CopyOp>(op);

        if (siblingCopy == nullptr || siblingCopy == srcCopyOp) {
            return false;
        }

        if (isClusterCopy && getClusterCopyResultType(srcClusterCopyOp) != getClusterCopyResultType(op)) {
            return false;
        }

        auto srcSubView = srcCopyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
        auto siblingSubView = siblingCopy.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
        if (isClusterCopy) {
            auto siblingClusterCopyOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(op);
            srcSubView = srcClusterCopyOp.getOutputs()[0].getDefiningOp<VPUIP::SubViewOp>();
            siblingSubView = siblingClusterCopyOp.getOutputs()[0].getDefiningOp<VPUIP::SubViewOp>();
        }

        if (srcSubView != nullptr && siblingSubView != nullptr && isSameSubViewFunc(srcSubView, siblingSubView)) {
            return true;
        }

        if (srcSubView == nullptr && siblingSubView == nullptr) {
            return true;
        }

        return false;
    };

    const auto updateSiblingCopyOutputBuff = [&](mlir::Operation* op) {
        // Get the sibling copy
        auto siblingCopy = isClusterCopy ? mlir::dyn_cast_or_null<VPUIP::CopyOp>(getCopyOpFromClusterTilingOp(op))
                                         : mlir::dyn_cast<VPUIP::CopyOp>(op);
        if (siblingCopy == nullptr) {
            nestedLogger.trace("Sibling op is not copy at {0}", op->getLoc());
            return;
        }
        // Get the buffer linked to copy output
        auto copyOpOutputBuff = VPUIP::getTopBufferOfNCEClusterTiling(copyOp, copyOp.getOutputBuff());
        // Get the buffer linked to sibling copy output that will be fused
        auto siblingCopyOutputBuff = VPUIP::getTopBufferOfNCEClusterTiling(siblingCopy, siblingCopy.getOutputBuff());
        // Replace the usage of sibling copy output buffer with copy output buffer
        siblingCopyOutputBuff.replaceAllUsesWith(copyOpOutputBuff);
    };

    auto parentOp = getParentOfCopyOp(copyOp);
    bool hasReplaceParallelCopies = false;
    // Optimize pattern: SubviewParentOp -> ParentOp(SubView) -> CopyOp/NCEClusterTiling(CopyOp)
    if (auto subViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(parentOp)) {
        if (auto subviewParentOp = subViewOp.getSource().getDefiningOp()) {
            for (auto* siblingOp : llvm::make_early_inc_range(subviewParentOp->getResult(0).getUsers())) {
                auto siblingSubViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(siblingOp);
                if (siblingSubViewOp == nullptr || !isSameSubViewFunc(subViewOp, siblingSubViewOp)) {
                    continue;
                }

                auto siblingCopyOp = *siblingSubViewOp.getResult().getUsers().begin();
                if (!isSameCopyFunc(copyOp, siblingCopyOp)) {
                    continue;
                }

                nestedLogger.trace("Fuse SubView op {0} to {1}", siblingSubViewOp->getLoc(), subViewOp->getLoc());
                updateSiblingCopyOutputBuff(siblingCopyOp);
                siblingSubViewOp.getOperation()->replaceAllUsesWith(subViewOp.getOperation());
                siblingCopyOp->replaceAllUsesWith(copyReplaceTarget);
                rewriter.eraseOp(siblingCopyOp);
                rewriter.eraseOp(siblingSubViewOp);
                hasReplaceParallelCopies = true;
            }
        }
    }

    // Optimize pattern: ParentOp -> CopyOp/NCEClusterTiling(CopyOp)
    for (auto* siblingOp : llvm::make_early_inc_range(parentOp->getResult(0).getUsers())) {
        if (isSameCopyFunc(copyOp, siblingOp)) {
            updateSiblingCopyOutputBuff(siblingOp);
            if (!isClusterCopy) {
                auto siblingCopy = mlir::dyn_cast<VPUIP::CopyOp>(siblingOp);
                nestedLogger.trace("Fuse Copy op {0} to {1}", siblingCopy->getLoc(), copyOp->getLoc());

                siblingCopy.getOperation()->replaceAllUsesWith(copyReplaceTarget);
            } else {
                auto siblingCluster = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(siblingOp);
                nestedLogger.trace("Fuse NCEClusterTiling Copy op {0} to {1}", siblingCluster->getLoc(),
                                   copyOp->getLoc());

                siblingCluster->replaceAllUsesWith(copyReplaceTarget);
            }
            rewriter.eraseOp(siblingOp);
            hasReplaceParallelCopies = true;
        }
    }

    return mlir::success(hasReplaceParallelCopies);
}

//
// OptimizeParallelCopiesPass
//

class OptimizeParallelCopiesPass final : public VPUIP::OptimizeParallelCopiesBase<OptimizeParallelCopiesPass> {
public:
    explicit OptimizeParallelCopiesPass(bool enableOptimizeConstCopy, Logger log)
            : _enableOptimizeConstCopy(enableOptimizeConstCopy) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    bool _enableOptimizeConstCopy;
};

// safeRunOnFunc

void OptimizeParallelCopiesPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ParallelCopiesRewriter>(&ctx, _log, _enableOptimizeConstCopy);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createOptimizeParallelCopiesPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createOptimizeParallelCopiesPass(bool enableOptimizeConstCopy, Logger log) {
    return std::make_unique<OptimizeParallelCopiesPass>(enableOptimizeConstCopy, log);
}
