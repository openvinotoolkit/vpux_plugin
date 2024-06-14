//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/expand_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

DimsOrder getNHWCOutputLayout(DimsOrder memPermute) {
    // To use NCE accelerate Permutation, we always cast the input tensor's layout to NHWC based on phyical layout.
    //  In this way, we only need consider the below 5 cases:
    //
    //                  NHWC (Case 0)
    //                   |
    //      NHCW  NWCH  NWHC  NCWH  NCHW
    // Case   1    2     3     4     5
    //
    const std::unordered_map<DimsOrder, DimsOrder> permuteToLayout = {{DimsOrder::NCWH, DimsOrder::NHCW},
                                                                      {DimsOrder::NHWC, DimsOrder::NWCH},
                                                                      {DimsOrder::NHCW, DimsOrder::NWHC},
                                                                      {DimsOrder::NWHC, DimsOrder::NCWH},
                                                                      {DimsOrder::NWCH, DimsOrder::NCHW}};
    const auto configIter = permuteToLayout.find(memPermute);
    VPUX_THROW_WHEN(configIter == permuteToLayout.end(), "The permute layout {0} not supported.", memPermute);
    return configIter->second;
}

bool isBeneficialToConvert(ShapeRef shape) {
    // If the MemPermute is legal to be converted to a pooling op. Need to compare with the DMA implementation.
    // Experimental data shows an linear correlation between inference time and permute data size for both ODU permute
    // and DMA permute with different slopes.
    // Experimental Constraint: utilize DMA conversion when data size is less than the threhold
    constexpr int64_t threshold = 32 * 16 * 224;
    return shape.totalSize() >= threshold;
}

SmallVector<std::pair<Shape, DimsOrder>> calculateConversions(ShapeRef originInputShape, int64_t alignedChannel,
                                                              DimsOrder targetOrder) {
    //
    //               NWCH (Case 2)
    //                 |
    //      NHCW  NWHC  NCWH  NCHW
    // Case   1    3     4     5
    //
    const std::unordered_map<DimsOrder, DimsOrder> dimHLayoutToPerm = {{DimsOrder::NHCW, DimsOrder::NWHC},
                                                                       {DimsOrder::NWHC, DimsOrder::NCWH},
                                                                       {DimsOrder::NCWH, DimsOrder::NHCW},
                                                                       {DimsOrder::NCHW, DimsOrder::NHWC}};

    //
    //          NCHW (Case 5)
    //             |
    //      NHCW  NWHC  NCWH
    // Case   1    3     4
    //
    const std::unordered_map<DimsOrder, DimsOrder> dimWLayoutToPerm = {{DimsOrder::NHCW, DimsOrder::NHCW},
                                                                       {DimsOrder::NWHC, DimsOrder::NWHC},
                                                                       {DimsOrder::NCWH, DimsOrder::NCWH}};

    bool dimHAligned = (originInputShape[Dims4D::Act::H] % alignedChannel) == 0;
    bool dimWAligned = (originInputShape[Dims4D::Act::W] % alignedChannel) == 0;
    bool dimWCAligned = ((originInputShape[Dims4D::Act::W] * originInputShape[Dims4D::Act::C]) % alignedChannel) == 0;
    bool dimHCAligned = ((originInputShape[Dims4D::Act::H] * originInputShape[Dims4D::Act::C]) % alignedChannel) == 0;
    SmallVector<std::pair<Shape, DimsOrder>> newMaxPoolOrder;

    auto getMaxPoolTargetDimOrder =
            [targetOrder](const std::unordered_map<DimsOrder, DimsOrder>& dimsLayoutToPermConfig) {
                const auto layoutPermute = dimsLayoutToPermConfig.find(targetOrder);
                VPUX_THROW_WHEN(layoutPermute == dimsLayoutToPermConfig.end(), "The layout should be considered.");
                return getNHWCOutputLayout(layoutPermute->second);
            };

    auto calculateSingleDimConversion = [&](bool mergedAlign, bool dimAligned, DimsOrder fromDimOrder,
                                            DimsOrder toDimOrder,
                                            const std::unordered_map<DimsOrder, DimsOrder>& layout2Perm) -> bool {
        if (!mergedAlign) {
            newMaxPoolOrder.clear();
            return false;  // Failed
        }
        Shape castShape = {
                originInputShape[fromDimOrder.dimAt(0)], alignedChannel, originInputShape[fromDimOrder.dimAt(1)],
                originInputShape[fromDimOrder.dimAt(2)] * originInputShape[fromDimOrder.dimAt(3)] / alignedChannel};

        newMaxPoolOrder.push_back({castShape, DimsOrder::NWCH});
        if (targetOrder == toDimOrder) {
            return false;
        }
        if (dimAligned) {
            castShape = {originInputShape[toDimOrder.dimAt(0)], originInputShape[toDimOrder.dimAt(3)],
                         originInputShape[toDimOrder.dimAt(1)], originInputShape[toDimOrder.dimAt(2)]};
            newMaxPoolOrder.push_back({castShape, getMaxPoolTargetDimOrder(layout2Perm)});
            return false;
        }
        return true;
    };

    auto needFollowProcess =
            calculateSingleDimConversion(dimWCAligned, dimHAligned, DimsOrder::NHWC, DimsOrder::NWCH, dimHLayoutToPerm);
    if (!needFollowProcess) {
        return newMaxPoolOrder;
    }
    needFollowProcess =
            calculateSingleDimConversion(dimHCAligned, dimWAligned, DimsOrder::NWCH, DimsOrder::NCHW, dimWLayoutToPerm);
    if (needFollowProcess) {
        // If need more process, the layout conversion will be like: NCHW -> NHWC.
        // And NHWC is input layout, so we can't convert this MemPermute to MaxPool.
        newMaxPoolOrder.clear();
    }
    return newMaxPoolOrder;
}

//
// MemPermuteRewriter
//

class MemPermuteRewriter final : public mlir::OpRewritePattern<IE::MemPermuteOp> {
public:
    MemPermuteRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MemPermuteOp>(ctx), _log(log) {
        this->setDebugName("MemPermuteRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::MemPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MemPermuteRewriter::matchAndRewrite(IE::MemPermuteOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto inOrder = DimsOrder::fromValue(origOp.getInput());
    const auto inShape = getShape(origOp.getInput());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);

    const auto elementType = origOp.getType().cast<NDTypeInterface>().getElementType();
    if (elementType.isInteger(8)) {
        return matchFailed(_log.nest(), rewriter, origOp, "MaxPool not support 8 bit integer");
    }
    if (inShape[Dim(0)] != 1) {
        return matchFailed(_log.nest(), rewriter, origOp, "MemPermuteOp with dim N > 1");
    }

    if (isTrivialPermute(inMemShape, origOp.getMemPerm())) {
        return matchFailed(_log.nest(), rewriter, origOp, "MemPermuteOp is actually a permute cast");
    }
    const auto memPerm = DimsOrder::fromAffineMap(origOp.getMemPerm());
    if (memPerm.dimAt(0) != Dims4D::Act::N) {
        return matchFailed(_log.nest(), rewriter, origOp, "MemPermuteOp with dim N changed dim position");
    }

    if (auto parentOp = origOp.getInput().getDefiningOp<IE::ExpandOp>()) {
        auto order = DimsOrder::fromValue(parentOp.getInput());
        if (!IE::isEligibleConvertToConv(parentOp, _log, getDebugName()) && parentOp->hasOneUse() &&
            order == DimsOrder::NCHW) {
            // For expand which will be lowered into DMA op, there is an optimization in another pass later which will
            // fuse pattern `input(NCHW) -> Expand -> Permute` into a single DMA op. So skip mempermute optimization
            // here.
            return matchFailed(_log.nest(), rewriter, origOp,
                               "MemPermuteOp will be fused with parent Expand op in later pass");
        }
    }
    if (memPerm == DimsOrder::NHCW && !isBeneficialToConvert(inShape)) {
        return matchFailed(_log.nest(), rewriter, origOp, "MemPermuteOp is not performant using OPU permute");
    }

    // Populate the target shape following NCHW order of dimensions.
    // Physical layout NHWC corresponds to logical layout NCHW.
    const Shape targetInShape = {inMemShape[MemDim(0)], inMemShape[MemDim(3)], inMemShape[MemDim(1)],
                                 inMemShape[MemDim(2)]};

    auto ctx = rewriter.getContext();

    const auto nhwcOrderAttr = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx));
    auto inLayoutCast = rewriter.create<IE::LayoutCastOp>(origOp.getLoc(), origOp.getInput(), nhwcOrderAttr);
    auto reshapedInput = rewriter.create<IE::ShapeCastOp>(origOp.getLoc(), inLayoutCast.getOutput(),
                                                          getIntArrayAttr(ctx, targetInShape));
    const auto& targetOrder = getNHWCOutputLayout(memPerm);

    const auto reshapeOutType = reshapedInput.getResult().getType().cast<NDTypeInterface>();
    const auto maxPoolOutType = reshapeOutType.changeDimsOrder(targetOrder);
    auto maxPool = IE::createIdentityMaxPool(reshapedInput.getResult(), maxPoolOutType, rewriter);
    auto alignInterface = mlir::dyn_cast_or_null<IE::AlignedChannelsOpInterface>(maxPool);
    VPUX_THROW_WHEN(alignInterface == nullptr, "{0} don't have aligninterface.", origOp);

    mlir::Value latestInput = maxPool->getResult(0);
    SmallVector<DimsOrder> newMaxPoolTargetOrder;
    if (alignInterface.verifyChannels().failed()) {
        const auto alignedChannel = alignInterface.getInputChannelAlignment();
        rewriter.eraseOp(maxPool);

        auto conversionMap = calculateConversions(targetInShape, alignedChannel, targetOrder);
        auto needSpillingWithMultiConversions =
                conversionMap.size() > 1 && maxPoolOutType.getTotalAllocSize() * 2 > vpux::VPU::getTotalCMXSize(origOp);
        auto isNotPerformant = memPerm == DimsOrder::NHCW && needSpillingWithMultiConversions;
        if (conversionMap.empty() || isNotPerformant) {
            rewriter.eraseOp(reshapedInput);
            rewriter.eraseOp(inLayoutCast);
            return matchFailed(_log.nest(), rewriter, origOp,
                               "Channels of an IE.MaxPool are not aligned or the Conversion is not performant.");
        }

        latestInput = reshapedInput.getResult();
        for (const auto& item : conversionMap) {
            auto shapeCastTmp = rewriter.createOrFold<IE::ShapeCastOp>(origOp.getLoc(), latestInput,
                                                                       getIntArrayAttr(ctx, item.first.raw()));
            const auto layoutCastType = shapeCastTmp.getType().cast<NDTypeInterface>();
            const auto outType = layoutCastType.changeDimsOrder(item.second);
            maxPool = IE::createIdentityMaxPool(shapeCastTmp, outType, rewriter);
            auto inLayoutCast =
                    rewriter.create<IE::LayoutCastOp>(origOp.getLoc(), maxPool->getResult(0), nhwcOrderAttr);
            latestInput = inLayoutCast.getOutput();
        }
    }

    const auto orderInAttr = mlir::AffineMapAttr::get(DimsOrder::fromValue(origOp.getOutput()).toAffineMap(ctx));
    auto outLayoutCast = rewriter.createOrFold<IE::LayoutCastOp>(origOp.getLoc(), latestInput, orderInAttr);

    _log.trace("Fuse {0} to {1}", origOp->getLoc(), maxPool->getLoc());

    const auto targetShape = getShape(origOp.getOutput()).raw();
    auto reshapedOut = rewriter.createOrFold<IE::ShapeCastOp>(origOp.getLoc(), origOp.getType(), outLayoutCast,
                                                              getIntArrayAttr(ctx, targetShape));
    rewriter.replaceOp(origOp, reshapedOut);

    return mlir::success();
}

//
// ConvertMemPermuteToPoolPass
//

class ConvertMemPermuteToPoolPass final : public IE::ConvertMemPermuteToPoolPassBase<ConvertMemPermuteToPoolPass> {
public:
    explicit ConvertMemPermuteToPoolPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ConvertMemPermuteToPoolPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MemPermuteRewriter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> IE::createConvertMemPermuteToPoolPass(Logger log) {
    return std::make_unique<ConvertMemPermuteToPoolPass>(log);
}
