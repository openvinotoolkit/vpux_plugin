//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/adjust_layout_utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// ReorderWithSubView
//

class ReorderWithSubView final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    ReorderWithSubView(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SliceOp>(ctx), _log(log) {
        setDebugName("ReorderWithSubView");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SliceOp origSubViewOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithSubView::matchAndRewrite(IE::SliceOp origSubViewOp,
                                                        mlir::PatternRewriter& rewriter) const {
    auto origReorderOp = origSubViewOp.source().getDefiningOp<IE::ReorderOp>();
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got reorder at '{0}' -> subview at '{1}' pair", origReorderOp->getLoc(), origSubViewOp->getLoc());

    if (!origReorderOp.getResult().hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, origSubViewOp, "Reorder has more then one user");
    }

    auto newSubViewOp =
            rewriter.create<IE::SliceOp>(origSubViewOp->getLoc(), origReorderOp.input(),
                                         origSubViewOp.static_offsetsAttr(), origSubViewOp.static_sizesAttr());

    rewriter.replaceOpWithNewOp<IE::ReorderOp>(origSubViewOp, newSubViewOp.result(), origReorderOp.dstOrderAttr());
    return mlir::success();
}

//
// ReorderWithExpand
//

class ReorderWithExpand final : public mlir::OpRewritePattern<IE::ExpandOp> {
public:
    ReorderWithExpand(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ExpandOp>(ctx), _log(log) {
        setDebugName("ReorderWithExpand");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ExpandOp origExpandOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

void swapExpandWithReorder(mlir::PatternRewriter& rewriter, IE::ExpandOp expandOp, IE::ReorderOp origReorderOp) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(expandOp);

    auto newExpandOp = rewriter.create<IE::ExpandOp>(expandOp->getLoc(), origReorderOp.input(),
                                                     expandOp.pads_beginAttr(), expandOp.pads_endAttr());

    rewriter.replaceOpWithNewOp<IE::ReorderOp>(expandOp, newExpandOp.output(), origReorderOp.dstOrderAttr());
}

mlir::LogicalResult ReorderWithExpand::matchAndRewrite(IE::ExpandOp origExpandOp,
                                                       mlir::PatternRewriter& rewriter) const {
    auto origReorderOp = origExpandOp.input().getDefiningOp<IE::ReorderOp>();
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got reorder at '{0}' -> Expand at '{1}' pair", origReorderOp->getLoc(), origExpandOp->getLoc());

    const auto isExpand = [](mlir::Operation* reorderUser) -> bool {
        return mlir::isa<IE::ExpandOp>(reorderUser);
    };

    if (!llvm::all_of(origReorderOp->getUsers(), isExpand)) {
        return matchFailed(_log.nest(), rewriter, origExpandOp,
                           "Reorder has more than one user and they are heterogeneous");
    }

    for (auto* reorderUser : llvm::make_early_inc_range(origReorderOp->getUsers())) {
        auto expandOp = mlir::cast<IE::ExpandOp>(reorderUser);
        swapExpandWithReorder(rewriter, expandOp, origReorderOp);
    }

    return mlir::success();
}

//
// ReorderWithSplit
//

class ReorderWithSplit final : public mlir::OpRewritePattern<IE::SplitOp> {
public:
    ReorderWithSplit(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SplitOp>(ctx), _log(log) {
        setDebugName("ReorderWithSplit");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SplitOp origSplitOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithSplit::matchAndRewrite(IE::SplitOp origSplitOp, mlir::PatternRewriter& rewriter) const {
    if (origSplitOp.axis() != nullptr) {
        return mlir::failure();
    }

    auto origReorderOp = origSplitOp.input().getDefiningOp<IE::ReorderOp>();
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got reorder at '{0}' -> Split at '{1}' pair", origReorderOp->getLoc(), origSplitOp->getLoc());

    const auto initialOrder = DimsOrder::fromValue(origReorderOp.input());

    SmallVector<IE::ReorderOp> outputReorders;
    outputReorders.reserve(origSplitOp.outputs().size());

    for (auto res : origSplitOp.outputs()) {
        if (!res.hasOneUse()) {
            return matchFailed(_log.nest(), rewriter, origSplitOp, "Split output #{0} has more then one user",
                               res.getResultNumber());
        }

        auto resReorderOp = mlir::dyn_cast<IE::ReorderOp>(*res.user_begin());
        if (resReorderOp == nullptr) {
            return matchFailed(_log.nest(), rewriter, origSplitOp, "Split output #{0} consumed by non Reorder",
                               res.getResultNumber());
        }

        const auto resOrder = DimsOrder::fromValue(resReorderOp.output());
        if (resOrder != initialOrder) {
            return matchFailed(_log.nest(), rewriter, origSplitOp, "Split output #{0} is reordered to different order",
                               res.getResultNumber());
        }

        outputReorders.push_back(resReorderOp);
    }

    auto newSplitOp = rewriter.create<IE::SplitOp>(origSplitOp->getLoc(), origReorderOp.input(), origSplitOp.axis(),
                                                   origSplitOp.num_splitsAttr(), origSplitOp.axis_valueAttr());

    for (auto ind : irange(outputReorders.size())) {
        auto oldResReorderOp = outputReorders[ind];
        auto newResVal = newSplitOp->getResult(checked_cast<uint32_t>(ind));
        rewriter.replaceOp(oldResReorderOp, newResVal);
    }

    return mlir::success();
}

//
// ReorderWithConcat
//

class ReorderWithConcat final : public mlir::OpRewritePattern<IE::ConcatOp> {
public:
    ReorderWithConcat(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConcatOp>(ctx), _log(log) {
        setDebugName("ReorderWithConcat");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConcatOp origConcatOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithConcat::matchAndRewrite(IE::ConcatOp origConcatOp,
                                                       mlir::PatternRewriter& rewriter) const {
    SmallVector<mlir::Value> initialInputs;
    initialInputs.reserve(origConcatOp.inputs().size());

    Optional<DimsOrder> initialOrder;

    for (auto arg : origConcatOp.inputs()) {
        auto argReorderOp = arg.getDefiningOp<IE::ReorderOp>();
        if (argReorderOp == nullptr) {
            return mlir::failure();
        }

        const auto argOrder = DimsOrder::fromValue(argReorderOp.input());
        if (!initialOrder.hasValue()) {
            initialOrder = argOrder;
        } else if (argOrder != initialOrder.getValue()) {
            return mlir::failure();
        }

        initialInputs.push_back(argReorderOp.input());
    }

    if (!initialOrder.hasValue()) {
        return mlir::failure();
    }

    const auto concatOrder = DimsOrder::fromValue(origConcatOp.inputs().front());

    auto newConcat = rewriter.create<IE::ConcatOp>(origConcatOp->getLoc(), initialInputs, origConcatOp.per_axisAttr(),
                                                   origConcatOp.static_offsetsAttr());

    rewriter.replaceOpWithNewOp<IE::ReorderOp>(origConcatOp, origConcatOp.getType(), newConcat.output(),
                                               concatOrder.toAffineMap(origConcatOp.getContext()));

    return mlir::success();
}

//
// ReorderWithQuantCast
//

class ReorderWithQuantCast final : public mlir::OpRewritePattern<IE::QuantizeCastOp> {
public:
    ReorderWithQuantCast(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::QuantizeCastOp>(ctx), _log(log) {
        setDebugName("ReorderWithQuantCast");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeCastOp origQuantCastOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithQuantCast::matchAndRewrite(IE::QuantizeCastOp origQuantCastOp,
                                                          mlir::PatternRewriter& rewriter) const {
    auto origReorderOp = origQuantCastOp.input().getDefiningOp<IE::ReorderOp>();
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    _log.trace("Got reorder at '{0}' -> quantize cast at '{1}' pair", origReorderOp->getLoc(),
               origQuantCastOp->getLoc());

    if (!origReorderOp.getResult().hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, origQuantCastOp, "Reorder has more then one user");
    }

    auto newQuantCastOp = rewriter.create<IE::QuantizeCastOp>(origQuantCastOp->getLoc(), origReorderOp.input(),
                                                              origQuantCastOp.dstElemTypeAttr());

    rewriter.replaceOpWithNewOp<IE::ReorderOp>(origQuantCastOp, newQuantCastOp.output(), origReorderOp.dstOrderAttr());
    return mlir::success();
}

//
// ReorderWithConvert
//

class ReorderWithConvert final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    ReorderWithConvert(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
        setDebugName("ReorderWithConvert");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp convertOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithConvert::matchAndRewrite(IE::ConvertOp convertOp,
                                                        mlir::PatternRewriter& rewriter) const {
    // Note that in this case we replace Convert -> Reorder with Reorder -> Convert
    // This is an opposite behavior, compared to other rewriters
    if (!convertOp.getResult().hasOneUse()) {
        return matchFailed(_log.nest(), rewriter, convertOp, "ConvertOp has more then one user");
    }

    auto origReorderOp = mlir::dyn_cast<IE::ReorderOp>(*convertOp.getResult().getUsers().begin());
    if (origReorderOp == nullptr) {
        return mlir::failure();
    }

    const auto srcType = convertOp.input().getType();
    const auto dstElemType = convertOp.dstElemType();
    if (getElemTypeSize(srcType) >= getElemTypeSize(dstElemType)) {
        return matchFailed(rewriter, convertOp, "Convert doesn't increase data size");
    }

    auto newReorderOp =
            rewriter.create<IE::ReorderOp>(origReorderOp->getLoc(), convertOp.input(), origReorderOp.dstOrderAttr());

    rewriter.replaceOpWithNewOp<IE::ConvertOp>(origReorderOp, origReorderOp.getType(), newReorderOp.output(),
                                               convertOp.dstElemTypeAttr());

    return mlir::success();
}

//
// ReorderWithLayer
//

class ReorderWithLayer final : public mlir::OpInterfaceRewritePattern<IE::LayoutInfoOpInterface> {
public:
    ReorderWithLayer(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<IE::LayoutInfoOpInterface>(ctx), _log(log) {
        setDebugName("ReorderWithLayer");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::LayoutInfoOpInterface layerOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReorderWithLayer::matchAndRewrite(IE::LayoutInfoOpInterface layerOp,
                                                      mlir::PatternRewriter& rewriter) const {
    if (mlir::isa<IE::ReorderOp>(layerOp)) {
        return mlir::failure();
    }

    _log.trace("Got layer operation '{0}' at '{1}'", layerOp->getName(), layerOp->getLoc());

    auto argReorderOp = layerOp->getOperand(0).getDefiningOp<IE::ReorderOp>();
    if (argReorderOp == nullptr) {
        return mlir::failure();
    }

    const auto propagatingOrder = DimsOrder::fromValue(argReorderOp.input());

    // Propagate first input layout and infer layout info
    auto orderInfo = layerOp.getLayoutInfo();
    orderInfo.setInput(0, propagatingOrder);
    layerOp.inferLayoutInfo(orderInfo);
    if (orderInfo.getInput(0) != propagatingOrder) {
        return matchFailed(_log.nest(), rewriter, layerOp, "Layer doesn't support propagating order {0}",
                           propagatingOrder);
    }

    // Check if additional reorders for other inputs are needed
    for (auto ind : irange<size_t>(1, orderInfo.getNumInputs())) {
        const auto input = layerOp->getOperand(checked_cast<uint32_t>(ind));
        const auto order = DimsOrder::fromValue(input);
        const auto isConstInput = mlir::isa_and_nonnull<Const::DeclareOp>(input.getDefiningOp());
        const auto isReorderInput = mlir::isa_and_nonnull<IE::ReorderOp>(input.getDefiningOp());

        if (order != orderInfo.getInput(ind) && !isConstInput && !isReorderInput) {
            return matchFailed(_log.nest(), rewriter, layerOp, "Non-constant inputs require additional Reorders");
        }
    }

    rewriter.startRootUpdate(layerOp);

    _log.nest(1).trace("Remove Reorder before the first input");
    layerOp->getOpOperand(0).set(argReorderOp.input());

    const auto inputs = layerOp->getOpOperands();
    for (auto i : irange<size_t>(1, inputs.size())) {
        auto& input = inputs[i];

        const auto curOrder = DimsOrder::fromValue(input.get());
        const auto supportedOrder = orderInfo.getInput(i);

        _log.nest(1).trace("Process input #{0}", i);
        if (curOrder != supportedOrder) {
            insertReorderForInput(layerOp, input, supportedOrder, rewriter, _log.nest());
        }
    }

    const auto outputs = layerOp->getOpResults();
    for (auto i : irange(outputs.size())) {
        auto output = outputs[i];

        const auto curOrder = DimsOrder::fromValue(output);
        const auto supportedOrder = orderInfo.getOutput(i);

        _log.nest(1).trace("Process output #{0}", i);
        if (curOrder != supportedOrder) {
            changeDimsOrder(output, supportedOrder, _log.nest());
            insertReorderForOutput(layerOp, output, curOrder, rewriter, _log.nest());
        }
    }

    rewriter.finalizeRootUpdate(layerOp);

    return mlir::success();
}

//
// OptimizeReordersPass
//

class OptimizeReordersPass final : public IE::OptimizeReordersBase<OptimizeReordersPass> {
public:
    explicit OptimizeReordersPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeReordersPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ReorderWithSubView>(&ctx, _log);
    patterns.add<ReorderWithExpand>(&ctx, _log);
    patterns.add<ReorderWithSplit>(&ctx, _log);
    patterns.add<ReorderWithConcat>(&ctx, _log);
    patterns.add<ReorderWithQuantCast>(&ctx, _log);
    patterns.add<ReorderWithLayer>(&ctx, _log);
    IE::ReorderOp::getCanonicalizationPatterns(patterns, &ctx);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        vpux::Logger::global().error("signalPassFailure");
        signalPassFailure();
    }

    mlir::RewritePatternSet cleanupPatterns(&ctx);
    cleanupPatterns.add<ReorderWithConvert>(&ctx, _log);
    IE::ReorderOp::getCanonicalizationPatterns(cleanupPatterns, &ctx);

    func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(cleanupPatterns),
                                                        getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeReordersPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeReordersPass(Logger log) {
    return std::make_unique<OptimizeReordersPass>(log);
}
