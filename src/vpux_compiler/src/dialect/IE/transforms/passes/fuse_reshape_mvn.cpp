//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// FuseReshapeMvn
//

class FuseReshapeMvn final : public mlir::OpRewritePattern<IE::MVNOp> {
public:
    FuseReshapeMvn(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MVNOp>(ctx), _log(log) {
        setDebugName("FuseReshapeMvn");
    }

    mlir::LogicalResult matchAndRewrite(IE::MVNOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseReshapeMvn::matchAndRewrite(IE::MVNOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto inType = origOp.getInput().getType().cast<NDTypeInterface>();

    if (inType.getDimsOrder() != DimsOrder::NHWC) {
        _log.trace("Only targeting NHWC layout.");
        return mlir::failure();
    }

    if (origOp.channelsFitIntoCMX()) {
        _log.trace("Only targeting large MVN decomposition cases.");
        return mlir::failure();
    }

    if (origOp.getInternalReshape().has_value()) {
        _log.trace("Op has already an internal reshape attr.");
        return mlir::failure();
    }

    // Input pattern (before MVN)
    auto reorder3 = origOp.getInput().getDefiningOp<IE::ReorderOp>();
    if (!reorder3) {
        _log.trace("Input pattern not found #1.");
        return mlir::failure();
    }
    auto reshape2 = reorder3.getInput().getDefiningOp<IE::ReshapeOp>();
    if (!reshape2) {
        _log.trace("Input pattern not found #2.");
        return mlir::failure();
    }
    auto reorder1 = reshape2.getInput().getDefiningOp<IE::ReorderOp>();
    if (!reorder1) {
        _log.trace("Input pattern not found #3.");
        return mlir::failure();
    }

    // Output pattern (after MVN)
    auto reorder5 = mlir::dyn_cast<IE::ReorderOp>(*(origOp.getOutput().getUsers().begin()));
    if (!reorder5) {
        _log.trace("Output pattern not found #1.");
        return mlir::failure();
    }
    // Last common op for the two patterns
    mlir::Operation* lastOp = reorder5.getOperation();
    IE::GroupConvolutionOp groupConv = nullptr;

    // After 'reorder5' check [optional] pattern
    if (auto affineReshape = mlir::dyn_cast<IE::AffineReshapeOp>(*lastOp->getResult(0).getUsers().begin())) {
        auto reorderPreGc = mlir::dyn_cast<IE::ReorderOp>(*(affineReshape.getOutput().getUsers().begin()));
        if (!reorderPreGc) {  // -> NHWC
            _log.trace("Optional GroupConv pattern not found #1.");
            return mlir::failure();
        }

        groupConv = mlir::dyn_cast<IE::GroupConvolutionOp>(*(reorderPreGc.getOutput().getUsers().begin()));
        if (!groupConv) {
            _log.trace("Optional GroupConv pattern not found #2.");
            return mlir::failure();
        }

        auto reorderPostGc = mlir::dyn_cast<IE::ReorderOp>(*(groupConv.getOutput().getUsers().begin()));
        if (!reorderPostGc) {  // -> NHWC
            _log.trace("Optional GroupConv pattern not found #3.");
            return mlir::failure();
        }
        lastOp = reorderPostGc;
    }

    // Back to common pattern
    auto reshape6 = mlir::dyn_cast<IE::ReshapeOp>(*lastOp->getResult(0).getUsers().begin());
    if (!reshape6) {
        _log.trace("Output pattern not found #2.");
        return mlir::failure();
    }

    auto reorder7 = mlir::dyn_cast<IE::ReorderOp>(*(reshape6.getOutput().getUsers().begin()));
    if (!reorder7) {
        _log.trace("Output pattern not found #3.");
        return mlir::failure();
    }

    auto oneUseChain = reorder3.getOutput().hasOneUse() && reshape2.getOutput().hasOneUse() &&
                       reorder1.getOutput().hasOneUse() && reorder5.getOutput().hasOneUse() &&
                       reshape6.getOutput().hasOneUse() && reorder7.getOutput().hasOneUse();
    if (!oneUseChain) {
        _log.trace("Multiple users found in the chain.");
        return mlir::failure();
    }

    const auto checkReorder = [](IE::ReorderOp op, DimsOrder iOrder, DimsOrder oOrder) -> bool {
        const auto iType = op.getInput().getType().cast<NDTypeInterface>();
        const auto oType = op.getOutput().getType().cast<NDTypeInterface>();
        return (iType.getDimsOrder() == iOrder) && (oType.getDimsOrder() == oOrder);
    };

    if (!checkReorder(reorder1, DimsOrder::NHWC, DimsOrder::NCHW) ||
        !checkReorder(reorder3, DimsOrder::NCHW, DimsOrder::NHWC) ||
        !checkReorder(reorder5, DimsOrder::NHWC, DimsOrder::NCHW) ||
        !checkReorder(reorder7, DimsOrder::NCHW, DimsOrder::NHWC)) {
        _log.trace("Unexpected Reorder i/o dims-order.");
        return mlir::failure();
    }

    // Check the entire chain in/out shape is the same
    const auto newInp = reorder1.getInput().getType().cast<NDTypeInterface>();
    const auto newOut = reorder7.getInput().getType().cast<NDTypeInterface>();
    if (newInp.getShape() != newOut.getShape()) {
        _log.trace("Mismatching i/o shapes.");
        return mlir::failure();
    }

    // Checks for C reshape value
    const auto inCh = newInp.getShape().raw()[Dims4D::Act::C.ind()];
    const auto reCh = inType.getShape().raw()[Dims4D::Act::C.ind()];
    const auto ratio = inCh / reCh;
    const auto simdFactor = 16;  // Shave detail using float16 (MVN1MeanVar impl)
    if ((inCh % reCh) || (inCh <= reCh) || (ratio > simdFactor) || (simdFactor % ratio)) {
        // Current limitation for max reshape factor = 16 comes from current Shave impl.
        // E.g. supported cases:
        //  512->32 ch means reshape factor = 512/32 = 16
        //  256->32 ch means reshape factor = 256/32 = 8
        _log.trace("Expecting in/out C reshape factor to be an divisor of {0}", simdFactor);
        return mlir::failure();
    }

    auto origShape = origOp.getOutput().getType().cast<NDTypeInterface>().getShape();
    auto internalReshapeAttr = getIntArrayAttr(rewriter.getContext(), origShape.toValues());

    if (!groupConv) {
        auto newMvnOp =
                rewriter.create<IE::MVNOp>(origOp->getLoc(), reorder1.getInput(), origOp.getAcrossChannelsAttr(),
                                           origOp.getNormalizeVarianceAttr(), origOp.getEpsAttr(), internalReshapeAttr);
        reorder7.replaceAllUsesWith(newMvnOp.getResult());
    } else {
        auto inFilter = groupConv.getFilter().getDefiningOp<Const::DeclareOp>();
        if (!inFilter) {
            _log.trace("GroupConvolution filter input not constant");
            return mlir::failure();
        }
        if (groupConv.getBias()) {
            _log.trace("GroupConvolution bias not implemented");
            return mlir::failure();
        }
        auto contentAttr = inFilter.getContentAttr();
        auto contentAttrType = contentAttr.getType();
        auto contentAttrShape = contentAttrType.getShape().raw();
        if ((contentAttrShape[Dims4D::Filter::IC.ind()] != 1) || (contentAttrShape[Dims4D::Filter::KX.ind()] != 1) ||
            (contentAttrShape[Dims4D::Filter::KY.ind()] != 1)) {
            _log.trace("GroupConvolution is not a per-channel Multiply");
            return mlir::failure();
        }

        auto gcInputType = groupConv.getInput().getType();
        SmallVector<int64_t> newFilterShape = {inCh, 1, 1, 1};
        auto elemType = gcInputType.cast<NDTypeInterface>().getElementType();

        const DimsOrder filterOrder = DimsOrder::NHWC;
        auto ctx = rewriter.getContext();
        auto newFilterAttr = vpux::getTensorAttr(filterOrder.toAffineMap(ctx), nullptr, nullptr);
        auto newFilterType = mlir::RankedTensorType::get(newFilterShape, elemType, newFilterAttr);
        const auto foldedAttr = contentAttr.cast<Const::ContentAttr>().fold();
        SmallVector<float> foldedVals = foldedAttr.getValues<float>();

        mlir::DenseElementsAttr newFoldedValsAttr;
        if (contentAttr.isSplat()) {
            newFoldedValsAttr = wrapData(newFilterType, foldedVals[0]);
        } else {
            int64_t repl = inCh / reCh;
            SmallVector<float> expandfoldedVals;
            for (int64_t i = 0; i < repl; i++) {
                expandfoldedVals.insert(expandfoldedVals.end(), foldedVals.begin(), foldedVals.end());
            }
            newFoldedValsAttr = wrapData(newFilterType, expandfoldedVals);
        }

        auto newConstOp = rewriter.create<vpux::Const::DeclareOp>(inFilter.getLoc(), newFilterType,
                                                                  Const::ContentAttr::get(newFoldedValsAttr));

        const auto padBeginAttr = groupConv.getPadsBeginAttr();
        const auto padEndAttr = groupConv.getPadsEndAttr();
        const auto dilationsAttr = groupConv.getDilationsAttr();
        const auto stridesAttr = groupConv.getStridesAttr();
        const auto groupAttr = getIntAttr(rewriter, inCh);

        auto newMvnOp =
                rewriter.create<IE::MVNOp>(origOp->getLoc(), reorder1.getInput(), origOp.getAcrossChannelsAttr(),
                                           origOp.getNormalizeVarianceAttr(), origOp.getEpsAttr(), internalReshapeAttr);
        auto newGroupConv =
                rewriter.create<IE::GroupConvolutionOp>(groupConv.getLoc(), newMvnOp.getResult(), newConstOp, nullptr,
                                                        stridesAttr, padBeginAttr, padEndAttr, dilationsAttr, groupAttr,
                                                        /*post_opAttr=*/nullptr, /*clampAttr=*/nullptr);

        reorder7.replaceAllUsesWith(newGroupConv.getResult());
    }

    return mlir::success();
}

//
// FuseReshapeMvnPass
//

class FuseReshapeMvnPass final : public IE::FuseReshapeMvnBase<FuseReshapeMvnPass> {
public:
    explicit FuseReshapeMvnPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void FuseReshapeMvnPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseReshapeMvn>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createFuseReshapeMvnPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseReshapeMvnPass(Logger log) {
    return std::make_unique<FuseReshapeMvnPass>(log);
}
