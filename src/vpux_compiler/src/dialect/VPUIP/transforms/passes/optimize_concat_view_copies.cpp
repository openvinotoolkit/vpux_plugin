//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"

using namespace vpux;

namespace {

//
// AvoidConcatExtraChannel
//
class AvoidConcatExtraChannel : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    AvoidConcatExtraChannel(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp copyOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::LogicalResult checkConcatUsers(mlir::Value output, int64_t& channels, int64_t& channelOffsets) const;
    mlir::LogicalResult checkConcatInputs(mlir::ValueRange inputs, int64_t& channels,
                                          SmallVector<VPUIP::NCEClusterTilingOp>& tilingCopies) const;

    mlir::Operation* createAlloc(mlir::PatternRewriter& rewriter, VPUIP::NCEClusterTilingOp copyOp,
                                 int64_t channels) const;

    Logger _log;
};

/*
 Check if all Concat users are Subview with same channels slice
 less than Concat channels
                          Concat (m output channels)
        |                                        |
    Subview (n output channels)              Subview (n output channels)
    m > n

*/
mlir::LogicalResult AvoidConcatExtraChannel::checkConcatUsers(mlir::Value output, int64_t& channels,
                                                              int64_t& channelOffsets) const {
    auto subviews = output.getUsers();
    if (subviews.empty()) {
        return mlir::failure();
    }

    for (const auto user : subviews) {
        auto subview = mlir::dyn_cast_or_null<VPUIP::SubViewOp>(user);

        if (subview == nullptr) {
            return mlir::failure();
        }

        auto offsets = parseIntArrayAttr<int64_t>(subview.getStaticOffsetsAttr());
        auto sizes = parseIntArrayAttr<int64_t>(subview.getStaticSizesAttr());

        if (subview.getStaticStrides().has_value()) {
            return mlir::failure();
        }

        if (channelOffsets != 0 && channelOffsets != offsets[Dims4D::Act::C.ind()]) {
            return mlir::failure();
        }

        if (channels != 0 && channels != sizes[Dims4D::Act::C.ind()]) {
            return mlir::failure();
        }

        channels = sizes[Dims4D::Act::C.ind()];
        channelOffsets = offsets[Dims4D::Act::C.ind()];
    }

    return mlir::success();
}

/*
 Check if all Concat inputs copy NCE result with more channels
 than Subview after Concat
 Concat joins its inputs not by channel dimension

            Op                              Op
            |                                 |
        TilingCopy (m output channels)     TilingCopy (m output channels)
            |                                 |
          Subview (m output channels)       Subview (m output channels)
            |                                 |
                                Concat (m output channels)
                                   |
                                Subview (n output channels)
    m > n

*/
mlir::LogicalResult AvoidConcatExtraChannel::checkConcatInputs(
        mlir::ValueRange inputs, int64_t& channels, SmallVector<VPUIP::NCEClusterTilingOp>& tilingCopies) const {
    if (inputs.empty()) {
        return mlir::failure();
    }

    for (auto input : inputs) {
        auto tilingCopy = input.getDefiningOp<VPUIP::NCEClusterTilingOp>();

        if (tilingCopy == nullptr || tilingCopy.getInnerTaskOpOfType<VPUIP::CopyOp>() == nullptr) {
            return mlir::failure();
        }

        if (!tilingCopy->getResult(0).hasOneUse()) {
            return mlir::failure();
        }

        auto copyOpOutput = tilingCopy.getOutputs()[0];

        auto subview = copyOpOutput.getDefiningOp<VPUIP::SubViewOp>();

        if (subview == nullptr) {
            return mlir::failure();
        }

        auto offsets = parseIntArrayAttr<int64_t>(subview.getStaticOffsetsAttr());
        auto sizes = parseIntArrayAttr<int64_t>(subview.getStaticSizesAttr());

        if (offsets[Dims4D::Act::C.ind()] != 0) {
            return mlir::failure();
        }

        if (channels >= sizes[Dims4D::Act::C.ind()]) {
            return mlir::failure();
        }

        if (VPUIP::getRootAlloc<mlir::memref::AllocOp>(subview.getSource()) == nullptr) {
            return mlir::failure();
        }

        tilingCopies.push_back(tilingCopy);
    }

    return mlir::success();
}

mlir::Operation* AvoidConcatExtraChannel::createAlloc(mlir::PatternRewriter& rewriter, VPUIP::NCEClusterTilingOp copyOp,
                                                      int64_t channels) const {
    auto copyOpOutput = copyOp.getOutputs()[0];

    auto subview = copyOpOutput.getDefiningOp<VPUIP::SubViewOp>();

    auto opOutputType = subview.getSource().getType().cast<vpux::NDTypeInterface>();
    auto sourceShape = opOutputType.getShape().toValues();
    sourceShape[Dims4D::Act::C] = channels;
    auto newOpOutputType = opOutputType.changeShape(ShapeRef(sourceShape));

    return allocateBuffersOfType(_log, copyOp->getLoc(), rewriter, newOpOutputType).front().getDefiningOp();
}

void recursivelyInferReturnTypes(VPUIP::SubViewOp subView) {
    for (auto child : subView.getResult().getUsers()) {
        if (auto childSubViewOp = mlir::dyn_cast<VPUIP::SubViewOp>(child)) {
            vpux::inferReturnTypes(childSubViewOp, vpux::InferShapedTypeMode::ALL);
            recursivelyInferReturnTypes(childSubViewOp);
        }
    }
}

/*

            Op                                Op
            |                                 |
        TilingCopy (m output channels)     TilingCopy (m output channels)
            |                                 |
          Subview (m output channels)       Subview (m output channels)
            |                                 |
                                Concat (m output channels)
                                   |
                                Subview (n output channels)
    m > n

    is converted to pattern

            Op (m output channels)           Op (m output channels)
            |                                 |
          Subview (n output channels)       Subview (n output channels)
            |                                 |
        TilingCopy (n output channels)     TilingCopy (n output channels)
            |                                 |
                                Concat (n output channels)

*/
mlir::LogicalResult AvoidConcatExtraChannel::matchAndRewrite(VPUIP::ConcatViewOp concatOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("Got VPUIP.ConcatViewOp at '{0}'", concatOp->getLoc());
    auto nestedLogger = _log.nest();

    auto concatOutput = concatOp.getOutput();
    int64_t numChannels = 0;
    int64_t channelOffsets = 0;
    if (checkConcatUsers(concatOutput, numChannels, channelOffsets).failed()) {
        nestedLogger.trace("Cannot optimize because of users requirements");
        return mlir::failure();
    }

    if (numChannels == 0 || numChannels >= getShape(concatOp.getOutput())[Dims4D::Act::C]) {
        nestedLogger.trace("Cannot optimize because of channels requirements");
        return mlir::failure();
    }

    auto inputs = concatOp.getInputs();
    SmallVector<VPUIP::NCEClusterTilingOp> inputsCopies;
    inputsCopies.reserve(inputs.size());
    if (checkConcatInputs(inputs, numChannels, inputsCopies).failed() || inputsCopies.empty()) {
        nestedLogger.trace("Cannot optimize because of input requirements");
        return mlir::failure();
    }

    SmallVector<int64_t> newOffset(concatOp.getType().cast<vpux::NDTypeInterface>().getRank(), 0);
    newOffset[Dims4D::Act::C.ind()] = channelOffsets;

    for (auto copy : inputsCopies) {
        auto copyOpInput = copy.getInputs()[0];

        if (auto distributedType = copyOpInput.getType().dyn_cast<VPUIP::DistributedBufferType>()) {
            const auto tileIndex = VPUIP::getTilingDimIndex(distributedType);
            if (tileIndex.has_value()) {
                auto tileIndexVal = tileIndex.value();
                if (!VPUIP::isChannelOffsetsAndTileDimCompatibleWithClusterCopy(newOffset, tileIndexVal,
                                                                                distributedType)) {
                    return mlir::failure();
                }
            }
        }
    }

    auto* alloc = createAlloc(rewriter, inputsCopies.front(), numChannels);

    if (alloc == nullptr) {
        nestedLogger.trace("Cannot allocate new buffer");
        return mlir::failure();
    }

    SmallVector<mlir::Value> concatInputs;
    concatInputs.reserve(inputs.size());

    for (auto copy : inputsCopies) {
        auto copyOpInput = copy.getInputs()[0];
        auto copyOpOutput = copy.getOutputs()[0];

        auto subview = copyOpOutput.getDefiningOp<VPUIP::SubViewOp>();

        auto sizes = parseIntArrayAttr<int64_t>(subview.getStaticSizesAttr());
        sizes[Dims4D::Act::C.ind()] = numChannels;

        SmallVector<int64_t> newSizes = to_small_vector(getShape(copyOpInput).raw());
        newSizes[Dims4D::Act::C.ind()] = numChannels;

        auto newSubviewNCE = rewriter.create<VPUIP::SubViewOp>(
                subview.getLoc(), copyOpInput, getIntArrayAttr(subview.getContext(), ArrayRef(newOffset)),
                getIntArrayAttr(subview.getContext(), ArrayRef(newSizes)));

        auto newSubviewCopy =
                rewriter.create<VPUIP::SubViewOp>(subview.getLoc(), alloc->getResult(0), subview.getStaticOffsetsAttr(),
                                                  getIntArrayAttr(subview.getContext(), ArrayRef(sizes)));

        const auto copyOutBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                            mlir::ValueRange newOperands) {
            builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
        };

        SmallVector<mlir::Value> inputsOutputOperands = {newSubviewNCE, newSubviewCopy};
        auto newTilingCopy = rewriter.create<VPUIP::NCEClusterTilingOp>(copy->getLoc(), newSubviewCopy.getType(),
                                                                        inputsOutputOperands, copyOutBodyBuilder);

        concatInputs.push_back(newTilingCopy.getResults()[0]);
    }

    auto newOp = rewriter.replaceOpWithNewOp<VPUIP::ConcatViewOp>(concatOp, concatInputs, alloc->getResult(0));

    for (auto user : newOp.getOutput().getUsers()) {
        if (auto subviewOp = mlir::dyn_cast<VPUIP::SubViewOp>(user)) {
            auto newOffsets = parseIntArrayAttr<int64_t>(subviewOp.getStaticOffsetsAttr());
            newOffsets[Dims4D::Act::C.ind()] = 0;
            auto newOffsetsAttr = getIntArrayAttr(subviewOp.getContext(), ArrayRef(newOffsets));
            subviewOp->setAttr(subviewOp.getStaticOffsetsAttrName(), newOffsetsAttr);
            vpux::inferReturnTypes(user, vpux::InferShapedTypeMode::ALL);

            recursivelyInferReturnTypes(subviewOp);
        }
    }

    for (auto copy : inputsCopies) {
        rewriter.eraseOp(copy);
    }

    nestedLogger.trace("Successfully optimized CMX->DDR->CMX pattern");
    return mlir::success();
}

//
// FuseConcatView
//

/*
    TilingCopyOp/CopyOp  ...  TilingCopyOp/CopyOp
               \                 /
                ConcatView (DDR)
                        |
                CopyOp(DDR2DDR)      TilingCopyOp/CopyOp
                        \              /
                        ConcatView (DDR)


    TilingCopyOp/CopyOp  ...  TilingCopyOp/CopyOp     TilingCopyOp/CopyOp
                     \                 |                  /
                                ConcatView (DDR)
*/

class FuseConcatView final : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    FuseConcatView(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
    }

    bool isLegalConcatViewPattern(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const;
    bool hasCopyOpForAllInputs(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const;
    bool hasOneDDR2DDRCopyWithConcatViewConsumer(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const;

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp concatViewOp, mlir::PatternRewriter& rewriter) const final;
    mlir::LogicalResult fuseTwoConcatViewInputs(VPUIP::ConcatViewOp concatViewOp, mlir::PatternRewriter& rewriter,
                                                vpux::Logger log) const;

private:
    Logger _log;
};

bool FuseConcatView::hasCopyOpForAllInputs(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const {
    log.nest().trace("Checking hasCopyOpForAllInputs");

    auto isCopyOpWithSingleUser = [&log](mlir::Operation* op) {
        if (auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(op)) {
            if (!mlir::isa<VPUIP::SubViewOp>(copyOp.getOutputBuff().getDefiningOp())) {
                log.nest().nest().trace("Parent CopyOp's output buffer is not defined by a SubViewOp: '{0}'",
                                        copyOp->getLoc());
                return false;
            }

            return copyOp.getOutput().hasOneUse();
        }

        if (auto clusterCopyOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(op)) {
            if (!mlir::isa<VPUIP::CopyOp>(clusterCopyOp.getInnerTaskOp())) {
                log.nest().nest().trace("ConcatView input is not Copy op: '{0}'", clusterCopyOp->getLoc());
                return false;
            }

            if (!mlir::isa<VPUIP::SubViewOp>(clusterCopyOp.getOutputBuffs()[0].getDefiningOp())) {
                log.nest().nest().trace(
                        "Parent NCEClusterTilingOp's output buffer is not defined by a SubViewOp: '{0}'",
                        clusterCopyOp->getLoc());
                return false;
            }

            return clusterCopyOp->hasOneUse();
        }

        return false;
    };

    return llvm::all_of(concatViewOp.getInputs(), [&](auto input) {
        return isCopyOpWithSingleUser(input.getDefiningOp());
    });
}

bool FuseConcatView::hasOneDDR2DDRCopyWithConcatViewConsumer(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const {
    log.nest().trace("Checking hasOneDDR2DDRCopyWithConcatViewConsumer");

    if (!concatViewOp.getOutput().hasOneUse()) {
        log.nest().nest().trace("ConcatView Op has more than one user");
        return false;
    }

    auto copyOp = mlir::dyn_cast<VPUIP::CopyOp>(*concatViewOp.getOutput().getUsers().begin());
    if (!copyOp) {
        log.nest().nest().trace("Consumer of ConcatView Op is not Copy Op");
        return false;
    }

    if (!copyOp.getOutput().hasOneUse()) {
        log.nest().nest().trace("CopyOp Op no user or has more than one user");
        return false;
    }

    if (!mlir::isa<VPUIP::ConcatViewOp>(*copyOp.getOutput().getUsers().begin())) {
        log.nest().nest().trace("Consumer of Copy Op is not ConcatView Op");
        return false;
    }

    return VPUIP::isCopyFromDDR(copyOp) && VPUIP::isCopyToDDR(copyOp);
}

// Fuse ConcatView Ops to remove unnecessary copies, two conditions need to be satisfied:
// a) The Stride Level for each ConcatView input (after fusing) should be no more than 2;
//     It's a runtime and HW limitation in order to get the right NNDMA descriptor, we support a maximum of 3D DMA
//     transfers with 2 levels of striding.
// b) The number of inputs from the second ConcatView, which come from the output of the first should no more than 1;
//     For example, first ConcatView has M inputs, second ConcatView has N inputs, out of which P of them are the output
//     of the first ConcatView After fusing, the number of input copies is: M * P + (N - P)
//     Can't ensure we get benefit when P is of a large size. Limit optimization to P=1.
bool FuseConcatView::isLegalConcatViewPattern(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const {
    if (concatViewOp.getOutput().use_empty()) {
        log.nest().trace("Cannot find user copy op at '{0}'", concatViewOp->getLoc());
        return false;
    }

    if (!hasCopyOpForAllInputs(concatViewOp, log)) {
        log.nest().trace("Not all inputs is CopyOp for first ConcatViewOp at '{0}'", concatViewOp->getLoc());
        return false;
    }

    if (!hasOneDDR2DDRCopyWithConcatViewConsumer(concatViewOp, log)) {
        log.nest().trace("Not only one user is DDR2DDR copy with ConcatViewOp for op at '{0}'", concatViewOp->getLoc());
        return false;
    }

    log.nest().trace("FuseConcatView: Found legal ConcatView pattern at op '{0}'", concatViewOp->getLoc());

    return true;
}

mlir::LogicalResult FuseConcatView::fuseTwoConcatViewInputs(VPUIP::ConcatViewOp concatViewOp,
                                                            mlir::PatternRewriter& rewriter, vpux::Logger log) const {
    // Get current concat's memref.alloc op, which will be removed
    auto firstConcatMemAlloc = VPUIP::getRootAlloc<mlir::memref::AllocOp>(concatViewOp.getOutputBuff());
    if (firstConcatMemAlloc == nullptr) {
        log.nest().trace("Cannot rewrite because current concat '{0}' output isn't master buffer",
                         concatViewOp->getLoc());
        return mlir::failure();
    }

    // Get Copy and next ConcatView Op
    auto outputCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(*concatViewOp.getOutput().getUsers().begin());
    VPUX_THROW_UNLESS(outputCopyOp != nullptr, "Cannot get DDR to DDR Copy Op after '{0}'", concatViewOp->getLoc());
    VPUIP::SubViewOp outCopySubView = outputCopyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();

    auto nextConcatViewOp = mlir::dyn_cast<VPUIP::ConcatViewOp>(*outputCopyOp.getOutput().getUsers().begin());
    VPUX_THROW_UNLESS(nextConcatViewOp != nullptr, "Cannot get second ConcatView Op");

    auto nextConcatMemAlloc = VPUIP::getRootAlloc<mlir::memref::AllocOp>(nextConcatViewOp.getOutputBuff());
    if (nextConcatMemAlloc == nullptr) {
        log.nest().trace("Cannot rewrite because next concat '{0}' output isn't master buffer",
                         nextConcatViewOp->getLoc());
        return mlir::failure();
    }

    // Create an array of the new input copy ops
    SmallVector<mlir::Value> newCopyInputs;
    SmallVector<mlir::Value> oldCopyInputs;
    SmallVector<VPUIP::SubViewOp> oldSubViewInputs;
    newCopyInputs.reserve(concatViewOp.getInputs().size() + nextConcatViewOp.getInputs().size() - 1);
    oldCopyInputs.reserve(concatViewOp.getInputs().size());
    oldSubViewInputs.reserve(concatViewOp.getInputs().size());

    auto isStrideConcat = [](VPUIP::SubViewOp subView) {
        if (subView.getStaticStridesAttr() == nullptr) {
            return false;
        }

        auto strides = parseIntArrayAttr<int64_t>(subView.getStaticStridesAttr());
        return llvm::any_of(strides, [](auto stride) {
            return stride > 1;
        });
    };

    for (size_t nextInIdx = 0; nextInIdx < nextConcatViewOp.getInputs().size(); ++nextInIdx) {
        auto siblingCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(nextConcatViewOp.getInputs()[nextInIdx].getDefiningOp());
        if (!(siblingCopyOp && siblingCopyOp == outputCopyOp)) {
            newCopyInputs.push_back(nextConcatViewOp.getInputs()[nextInIdx]);
            continue;
        }

        SmallVector<int64_t> outCopyOffsets = parseIntArrayAttr<int64_t>(outCopySubView.getStaticOffsetsAttr());
        SmallVector<int64_t> outCopySizes = parseIntArrayAttr<int64_t>(outCopySubView.getStaticSizesAttr());
        if (isStrideConcat(outCopySubView)) {
            log.nest().trace("Fusing Concat Op with stride has no performance benefits");
            return mlir::failure();
        }

        for (size_t firstInIdx = 0; firstInIdx < concatViewOp.getInputs().size(); ++firstInIdx) {
            auto op = concatViewOp.getInputs()[firstInIdx].getDefiningOp();

            bool isClusterCopy = false;
            VPUIP::SubViewOp inCopySubView;
            auto inCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(op);
            if (inCopyOp) {
                inCopySubView = inCopyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
            }
            auto clusterCopyOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(op);
            if (clusterCopyOp) {
                inCopySubView = clusterCopyOp.getOutputBuffs()[0].getDefiningOp<VPUIP::SubViewOp>();
                isClusterCopy = true;
            }

            VPUX_THROW_WHEN(inCopySubView == nullptr, "Cannot get SubViewOp");
            oldCopyInputs.push_back(concatViewOp.getInputs()[firstInIdx]);
            oldSubViewInputs.push_back(inCopySubView);

            SmallVector<int64_t> inCopyOffsets = parseIntArrayAttr<int64_t>(inCopySubView.getStaticOffsetsAttr());
            SmallVector<int64_t> inCopySizes = parseIntArrayAttr<int64_t>(inCopySubView.getStaticSizesAttr());

            VPUX_THROW_WHEN(outCopyOffsets.size() != inCopyOffsets.size() || outCopySizes.size() != inCopySizes.size(),
                            "Input and output copy subviews have different-sized attributes");

            SmallVector<int64_t> newCopyOffsets(outCopyOffsets.size());
            SmallVector<int64_t> newCopySizes(outCopySizes.size());

            SmallVector<int64_t> newCopyStrides(inCopyOffsets.size(), 1);
            auto inCopyStrides = inCopySubView.getStaticStridesAttr();
            if (inCopyStrides != nullptr) {
                newCopyStrides = parseIntArrayAttr<int64_t>(inCopyStrides);
            }

            for (size_t idx = 0; idx < newCopyOffsets.size(); ++idx) {
                newCopySizes[idx] = inCopySizes[idx];
                newCopyOffsets[idx] = outCopyOffsets[idx] + inCopyOffsets[idx];
            }

            auto newSubViewOp = rewriter.create<VPUIP::SubViewOp>(outCopySubView->getLoc(), outCopySubView.getSource(),
                                                                  newCopyOffsets, newCopySizes, newCopyStrides);
            if (newSubViewOp->isBeforeInBlock(nextConcatMemAlloc)) {
                nextConcatMemAlloc->moveBefore(newSubViewOp);
            }

            if (isClusterCopy) {
                const auto copyOutBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                                    mlir::ValueRange newOperands) {
                    builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
                };

                SmallVector<mlir::Value> inputOutputOperands = {op->getOperand(0), newSubViewOp.getResult()};
                auto newCopyInCluster = rewriter.create<VPUIP::NCEClusterTilingOp>(
                        op->getLoc(), newSubViewOp->getResult(0).getType(), inputOutputOperands, copyOutBodyBuilder);

                auto innerCopyOp = newCopyInCluster.getInnerTaskOpOfType<VPUIP::CopyOp>();
                if (!VPUIP::hasLegalStridingLevel(innerCopyOp)) {
                    log.nest().trace("DMA Striding Level is illegal. Fusing Concat Op have no benefit");
                    rewriter.eraseOp(newCopyInCluster);
                    rewriter.eraseOp(newSubViewOp);
                    return mlir::failure();
                }
                newCopyInputs.push_back(newCopyInCluster->getResult(0));
                continue;
            }

            auto newCopyOp = rewriter.create<VPUIP::CopyOp>(op->getLoc(), op->getOperand(0), newSubViewOp.getResult());
            if (!VPUIP::hasLegalStridingLevel(newCopyOp)) {
                log.nest().trace("DMA Striding Level is illegal. Fusing Concat Op have no benefit");
                rewriter.eraseOp(newCopyOp);
                rewriter.eraseOp(newSubViewOp);
                return mlir::failure();
            }
            newCopyInputs.push_back(newCopyOp.getOutput());
        }
    }

    rewriter.setInsertionPoint(nextConcatViewOp);
    rewriter.replaceOpWithNewOp<VPUIP::ConcatViewOp>(nextConcatViewOp, nextConcatViewOp.getOutput().getType(),
                                                     newCopyInputs, nextConcatViewOp.getOutputBuff());

    // Erase the old hanging structure
    rewriter.eraseOp(outputCopyOp);
    rewriter.eraseOp(outCopySubView);
    rewriter.eraseOp(concatViewOp);

    for (size_t inIdx = 0; inIdx < oldCopyInputs.size(); ++inIdx) {
        rewriter.eraseOp(oldCopyInputs[inIdx].getDefiningOp());
        rewriter.eraseOp(oldSubViewInputs[inIdx]);
    }

    rewriter.eraseOp(firstConcatMemAlloc);

    return mlir::success();
}

mlir::LogicalResult FuseConcatView::matchAndRewrite(VPUIP::ConcatViewOp concatViewOp,
                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("FuseConcatView: Got ConcatView Op at '{0}'", concatViewOp.getLoc());

    if (!isLegalConcatViewPattern(concatViewOp, _log)) {
        _log.nest().trace("FuseConcatView: Cannot rewrite this concat Op");
        return mlir::failure();
    }

    return fuseTwoConcatViewInputs(concatViewOp, rewriter, _log);
}

// RemoveDDRToDDRCopyAfterConcatView
//

/*
            CopyOp     ...      CopyOp
               \                 /
                ConcatView (DDR)
                        |
                (Pure View Ops)
                        |
                CopyOp(DDR2DDR)

Optimized:
            CopyOp     ...      CopyOp
               \                 /
                ConcatView (DDR)
                        |
                (Pure View Ops)
*/

class RemoveDDRToDDRCopyAfterConcatView final : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    RemoveDDRToDDRCopyAfterConcatView(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
    }

    mlir::Operation* getTargetCopyOp(VPUIP::ConcatViewOp concatViewOp, vpux::Logger log) const;

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp concatViewOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::Operation* RemoveDDRToDDRCopyAfterConcatView::getTargetCopyOp(VPUIP::ConcatViewOp concatViewOp,
                                                                    vpux::Logger log) const {
    log.nest().trace("Checking ConcatView Copy pattern");

    auto childOp = *concatViewOp.getOutput().getUsers().begin();
    while (childOp != nullptr && mlir::isa<VPUIP::GenericReshapeOp, VPUIP::PermuteCastOp, VPUIP::QuantizeCastOp,
                                           VPUIP::ShapeCastOp, VPUIP::CopyOp>(childOp)) {
        if (!childOp->getResult(0).hasOneUse()) {
            log.nest().trace("child op user does not match");
            return nullptr;
        } else if (mlir::isa<VPUIP::CopyOp>(childOp)) {
            return childOp;
        } else {
            childOp = *childOp->getResult(0).getUsers().begin();
        }
    }
    log.nest().trace("Could not find ConcatView Copy pattern");
    return nullptr;
}

mlir::LogicalResult RemoveDDRToDDRCopyAfterConcatView::matchAndRewrite(VPUIP::ConcatViewOp concatViewOp,
                                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("RemoveDDRToDDRCopyAfterConcatView: Got ConcatView Op at '{0}'", concatViewOp.getLoc());

    if (!concatViewOp.getOutput().hasOneUse()) {
        _log.nest().trace("RemoveDDRToDDRCopyAfterConcatView: Only support ConcatView has one user");
        return mlir::failure();
    }
    auto targetOp = getTargetCopyOp(concatViewOp, _log);
    if (targetOp == nullptr) {
        _log.nest().trace("RemoveDDRToDDRCopyAfterConcatView: Cannot find the target Copy Op");
        return mlir::failure();
    }
    auto targetCopyOp = mlir::dyn_cast<VPUIP::CopyOp>(targetOp);
    if (!VPUIP::isCopyToDDR(targetCopyOp) || !VPUIP::isCopyFromDDR(targetCopyOp)) {
        _log.nest().trace("RemoveDDRToDDRCopyAfterConcatView: Target Copy Op is not from DDR to DDR");
        return mlir::failure();
    }

    // Check if the CopyOp copies to output
    if (targetCopyOp.getOutputBuff().isa<mlir::BlockArgument>()) {
        _log.trace("RemoveDDRToDDRCopyAfterConcatView: Cannot rewrite because it is last copy");
        return mlir::failure();
    }

    VPUIP::SubViewOp outCopySubView = targetCopyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
    if (outCopySubView != nullptr) {
        _log.nest().trace("Cannot remove copy op with subView");
        return mlir::failure();
    }

    targetCopyOp.getOutput().replaceAllUsesWith(targetCopyOp.getInput());
    rewriter.eraseOp(targetCopyOp);
    _log.trace("Successfully removed redundant copy Op after ConcatView");
    return mlir::success();
}

//
// MoveConcatViewWithClusteredCopyToCMX
//

/*
    Move ConcatView from DDR to CMX when inputs and output ClusterTilingCopy is Duplicated.
    TODO: Support more case when ConcatView has non-clusterd CopyOp user, see E#102977

    Convert below pattern:

      ClusterTilingCopy  ...     CopyOp
        (CMX -> DDR)          (DDR -> DDR)
               \                /
                ConcatView (DDR)
                        |
                (Pure View Ops)
                        |
                ClusterTilingCopy
                   (DDR -> CMX)
                        |

    to:

      ClusterTilingCopy
        (CMX -> DDR)
             |
          AllocOp (DDR)
             |
      ClusterTilingCopy  ...  ClusterTilingCopy
        (DDR -> CMX)            (DDR -> CMX)
               \                /
                ConcatView (CMX)
                        |
                (Pure View Ops) (CMX)
                        |
                DistributedCast
                        |

    So that DDR2DDR copy inputs can be optimized.
*/

struct ConcatInputs {
    SmallVector<mlir::Value> inputCopies;
    SmallVector<mlir::Value> inputClusterCopies;
};

struct ConcatOutputs {
    SmallVector<mlir::Operation*> viewLikeOps;
    VPUIP::NCEClusterTilingOp outputClusterCopy;
};

class MoveConcatViewWithClusteredCopyToCMX final : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    MoveConcatViewWithClusteredCopyToCMX(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
        setDebugName("MoveConcatViewWithClusteredCopyToCMX");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp concatViewOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;

    mlir::FailureOr<mlir::Operation*> searchCopyOpThroughViewLikeOps(VPUIP::ConcatViewOp concatViewOp,
                                                                     SmallVector<mlir::Operation*>& viewLikeOps) const;

    mlir::FailureOr<ConcatInputs> getValidConcatInputs(VPUIP::ConcatViewOp concatViewOp) const;
    mlir::FailureOr<ConcatOutputs> getValidConcatOutputs(VPUIP::ConcatViewOp concatViewOp) const;

    void convertCopyInputAndStore(ArrayRef<mlir::Value> inputCopies, mlir::Value outputBuffer,
                                  SmallVector<mlir::Value>& newConcatInputs, mlir::PatternRewriter& rewriter) const;
    void convertClusterTilingCopyInputAndStore(ArrayRef<mlir::Value> inputClusterCopies, mlir::Value outputBuffer,
                                               SmallVector<mlir::Value>& newConcatInputs,
                                               mlir::PatternRewriter& rewriter) const;

    VPUIP::DistributedBufferType getDuplicatedDistributedType(NDTypeInterface ndType,
                                                              VPUIP::DistributedBufferType distributedType,
                                                              mlir::MLIRContext* ctx) const;
    mlir::Value rewriteViewLikeOps(mlir::Value input, ArrayRef<mlir::Operation*> viewLikeOps,
                                   VPUIP::DistributedBufferType origOutputBufferType,
                                   mlir::PatternRewriter& rewriter) const;
};

VPUIP::DistributedBufferType MoveConcatViewWithClusteredCopyToCMX::getDuplicatedDistributedType(
        NDTypeInterface ndType, VPUIP::DistributedBufferType distributedType, mlir::MLIRContext* ctx) const {
    const auto orderMap = mlir::AffineMapAttr::get(ndType.getDimsOrder().toAffineMap(ctx));
    const auto shape = ndType.getShape();
    const auto elemType = ndType.getElementType();

    auto distribution = distributedType.getDistribution();
    auto memSpace = distributedType.getMemSpace();

    if (VPU::isDistributedAttrWithExplicitShapesAndOffsets(distribution)) {
        VPUX_THROW_WHEN(distribution.getMode().getValue() != VPU::DistributionMode::DUPLICATED,
                        "DistributedBufferType is not DUPLICATED, type = {0}", distributedType);

        auto newDistribution = VPU::getNonOverlappedDistributedAttr(shape, distribution.getMode(), nullptr,
                                                                    distribution.getNumClusters(), nullptr,
                                                                    distribution.getUniformDistributedSegments(), ctx);

        return VPUIP::DistributedBufferType::get(ctx, shape.raw(), elemType, orderMap, memSpace, newDistribution);
    }

    auto newDistribution = VPU::DistributedTensorAttr::get(
            ctx, distribution.getMode(), distribution.getNumTiles(), nullptr, nullptr, nullptr,
            distribution.getNumClusters(), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    return VPUIP::DistributedBufferType::get(ctx, shape.raw(), elemType, orderMap, memSpace, newDistribution);
};

bool isDuplicated(VPU::DistributionMode mode) {
    return VPU::bitEnumContainsAny(mode, VPU::DistributionMode::DUPLICATED) ||
           VPU::bitEnumContainsAny(mode, VPU::DistributionMode::MULTICASTED);
}

// Check inputs of ConcatView, below pattern is expected.
//   ClusterTilingCopy  ...     CopyOp
//      (CMX -> DDR)          (DDR -> DDR)
//             \                /
//              ConcatView (DDR)
// Pattern matching requires below criteria:
// 1.If ConcatView has ClusterTilingCopy inputs, they should be DUPLICATED.
// 2.ConcatView should have at least one DDR2DDR copy input.
// Return ConcatInputs struct if pattern can match, otherwise return mlir::failure().
mlir::FailureOr<ConcatInputs> MoveConcatViewWithClusteredCopyToCMX::getValidConcatInputs(
        VPUIP::ConcatViewOp concatViewOp) const {
    const auto isDDR2DDRCopy = [](mlir::Value input) {
        auto op = mlir::dyn_cast_or_null<VPUIP::CopyOp>(input.getDefiningOp());
        if (op == nullptr) {
            return false;
        }

        // check if output buff is a SubView for safety
        auto subViewOp = op.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
        if (subViewOp == nullptr) {
            return false;
        }

        return VPUIP::isCopyToDDR(op) && VPUIP::isCopyFromDDR(op);
    };

    const auto isDuplicatedClusterCopy = [](mlir::Value input) {
        auto clusterOp = mlir::dyn_cast_or_null<VPUIP::NCEClusterTilingOp>(input.getDefiningOp());
        if (clusterOp == nullptr) {
            return false;
        }

        auto innerOp = clusterOp.getInnerTaskOpOfType<VPUIP::CopyOp>();
        if (innerOp == nullptr || !VPUIP::isCopyToDDR(innerOp)) {
            return false;
        }

        // check if output buff is a SubView for safety
        auto subViewOp = clusterOp.getOutputBuffs()[0].getDefiningOp<VPUIP::SubViewOp>();
        if (subViewOp == nullptr) {
            return false;
        }

        auto tilingCopyInput = clusterOp->getOperand(0);
        const auto inDistributedType = VPUIP::extractDataType(tilingCopyInput).dyn_cast<VPUIP::DistributedBufferType>();
        VPUX_THROW_UNLESS(inDistributedType != nullptr, "Cannot get distributedType");

        auto distribution = inDistributedType.getDistribution();
        return isDuplicated(distribution.getMode().getValue());
    };

    struct ConcatInputs validInputs;

    for (const auto& input : concatViewOp.getInputs()) {
        if (isDDR2DDRCopy(input)) {
            validInputs.inputCopies.push_back(input);
        } else if (isDuplicatedClusterCopy(input)) {
            validInputs.inputClusterCopies.push_back(input);
        } else {
            _log.nest().trace("[{0}] Invalid input: not a valid Copy", getDebugName());
            return mlir::failure();
        }
    }

    if (validInputs.inputCopies.empty()) {
        _log.nest().trace("[{0}] Invalid input: not DDR2DDR Copy input", getDebugName());
        return mlir::failure();
    }

    return validInputs;
}

// Traverse output chain, store pure viewlike ops into viewLikeOps vector and return ClusterTilingCopy.
// Return mlir::failure() if pattern does not match
mlir::FailureOr<mlir::Operation*> MoveConcatViewWithClusteredCopyToCMX::searchCopyOpThroughViewLikeOps(
        VPUIP::ConcatViewOp concatViewOp, SmallVector<mlir::Operation*>& viewLikeOps) const {
    auto isClusterTilingCopyOp = [](mlir::Operation* user) {
        if (auto tilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(user)) {
            return tilingOp.getInnerTaskOpOfType<VPUIP::CopyOp>() != nullptr;
        }
        return false;
    };

    auto isSupportedViewlikeOp = [](mlir::Operation* user) {
        return mlir::isa<VPUIP::PermuteCastOp, VPUIP::GenericReshapeOp, VPUIP::ShapeCastOp>(user);
    };

    mlir::Operation* operation = concatViewOp;
    while (operation && !operation->getUsers().empty()) {
        auto user = *(operation->getUsers().begin());

        if (isClusterTilingCopyOp(user)) {
            return user;
        } else if (isSupportedViewlikeOp(user)) {
            if (!user->hasOneUse()) {
                return mlir::failure();
            }
            viewLikeOps.push_back(user);
            operation = user;
            continue;
        } else {
            break;
        }
    }
    return mlir::failure();
}

// Check ConcatView output chain.
// We expect ConcatView is followed by several viewlike ops(optional), and then a DUPLICATED ClusterTilingCopy is
// connected. Like in below:
//      ConcatView
//          |
//    (Pure View Ops)
//          |
//   ClusterTilingCopy
//          |
// Return ConcatOutputs struct if pattern can match, otherwise return mlir::failure().
mlir::FailureOr<ConcatOutputs> MoveConcatViewWithClusteredCopyToCMX::getValidConcatOutputs(
        VPUIP::ConcatViewOp concatViewOp) const {
    struct ConcatOutputs validOutput;

    auto copyAfterViewLikeOps = searchCopyOpThroughViewLikeOps(concatViewOp, validOutput.viewLikeOps);
    if (mlir::failed(copyAfterViewLikeOps)) {
        _log.nest().trace("[{0}] Invalid output: no CopyOp after viewlike ops", getDebugName());
        return mlir::failure();
    }

    const auto isDuplicatedChildClusterCopyOp = [](mlir::Operation* op) {
        auto clusterOp = mlir::dyn_cast_or_null<VPUIP::NCEClusterTilingOp>(op);
        if (clusterOp == nullptr) {
            return false;
        }

        auto innerOp = clusterOp.getInnerTaskOpOfType<VPUIP::CopyOp>();
        if (innerOp == nullptr || !VPUIP::isCopyFromDDR(innerOp) || VPUIP::isCopyToDDR(innerOp)) {
            return false;
        }

        auto tilingCopyOutput = clusterOp->getResult(0);
        const auto outputDistributedType =
                VPUIP::extractDataType(tilingCopyOutput).dyn_cast<VPUIP::DistributedBufferType>();
        VPUX_THROW_UNLESS(outputDistributedType != nullptr, "Cannot get distributedType");

        auto distribution = outputDistributedType.getDistribution();
        return isDuplicated(distribution.getMode().getValue());
    };

    auto childOp = copyAfterViewLikeOps.value();
    if (!isDuplicatedChildClusterCopyOp(childOp)) {
        _log.nest().trace("[{0}] Invalid output: no duplicated cluster CopyOp", getDebugName());
        return mlir::failure();
    }

    auto clusterCopyOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(childOp);
    auto outputBuffer = clusterCopyOp.getOutputBuffs()[0];
    auto masterBuffer = VPUIP::getRootAlloc<VPURT::AllocDistributed>(outputBuffer);
    if (masterBuffer == nullptr) {
        _log.nest().trace("[{0}] Invalid output: buffer isn't master buffer", getDebugName());
        return mlir::failure();
    }

    validOutput.outputClusterCopy = clusterCopyOp;

    return validOutput;
}

void MoveConcatViewWithClusteredCopyToCMX::convertCopyInputAndStore(ArrayRef<mlir::Value> inputCopies,
                                                                    mlir::Value outputBuffer,
                                                                    SmallVector<mlir::Value>& newConcatInputs,
                                                                    mlir::PatternRewriter& rewriter) const {
    for (const auto& copyInput : inputCopies) {
        auto inputCopyOp = copyInput.getDefiningOp<VPUIP::CopyOp>();
        auto subViewOp = inputCopyOp.getOutputBuff().getDefiningOp<VPUIP::SubViewOp>();
        VPUX_THROW_WHEN(subViewOp == nullptr, "Can't find SubViewOp");
        auto newSubView = rewriter.create<VPUIP::SubViewOp>(
                appendLoc(subViewOp->getLoc(), "_subview_CMX"), outputBuffer, subViewOp.getStaticOffsetsAttr(),
                subViewOp.getStaticSizesAttr(), subViewOp.getStaticStridesAttr());

        const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location, mlir::ValueRange newOperands) {
            builder.create<VPUIP::CopyOp>(inputCopyOp->getLoc(), newOperands[0], newOperands[1]);
        };
        auto inputsOutputOperands = {inputCopyOp.getInput(), newSubView.getResult()};
        auto newClusterTilingOutType = newSubView.getResult().getType().cast<vpux::NDTypeInterface>();
        auto newClusterTilingCopyOp =
                rewriter.create<VPUIP::NCEClusterTilingOp>(appendLoc(inputCopyOp->getLoc(), "_cvt_from_copy_input"),
                                                           newClusterTilingOutType, inputsOutputOperands, bodyBuilder);

        // remove old CopyOp
        rewriter.replaceOp(inputCopyOp, newClusterTilingCopyOp->getResult(0));

        newConcatInputs.push_back(newClusterTilingCopyOp.getResults()[0]);
    }
}

void MoveConcatViewWithClusteredCopyToCMX::convertClusterTilingCopyInputAndStore(
        ArrayRef<mlir::Value> inputClusterCopies, mlir::Value outputBuffer, SmallVector<mlir::Value>& newConcatInputs,
        mlir::PatternRewriter& rewriter) const {
    for (const auto& clusterCopyInput : inputClusterCopies) {
        auto inputClusterCopyOp = clusterCopyInput.getDefiningOp<VPUIP::NCEClusterTilingOp>();
        auto subViewOp = inputClusterCopyOp.getOutputBuffs()[0].getDefiningOp<VPUIP::SubViewOp>();
        VPUX_THROW_WHEN(subViewOp == nullptr, "Can't find SubViewOp");

        // Input data need copy to DDR then copy back to CMX since ClusterTilingCopy from DistributedBufferType to
        // DistributedBufferType is not supported

        // CMX to DDR
        auto inputType = inputClusterCopyOp.getInnerTaskOp()->getOperand(0).getType().dyn_cast<vpux::NDTypeInterface>();
        auto newDDRType = inputType.changeMemSpace(VPU::MemoryKind::DDR);
        auto newAllocDDROp = rewriter.create<mlir::memref::AllocOp>(
                appendLoc(inputClusterCopyOp->getLoc(), "_new_DDR_buffer"), newDDRType.cast<mlir::MemRefType>());

        const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location, mlir::ValueRange newOperands) {
            builder.create<VPUIP::CopyOp>(inputClusterCopyOp->getLoc(), newOperands[0], newOperands[1]);
        };
        auto inputsOutputOperands = {inputClusterCopyOp->getOperand(0), static_cast<mlir::Value>(newAllocDDROp)};
        auto cmxToDDRClusterTilingCopyOp =
                rewriter.create<VPUIP::NCEClusterTilingOp>(appendLoc(inputClusterCopyOp->getLoc(), "_CMX_to_DDR_Copy"),
                                                           newDDRType, inputsOutputOperands, bodyBuilder);

        // DDR to CMX
        auto newSubView = rewriter.create<VPUIP::SubViewOp>(
                appendLoc(subViewOp->getLoc(), "_subview_CMX"), outputBuffer, subViewOp.getStaticOffsetsAttr(),
                subViewOp.getStaticSizesAttr(), subViewOp.getStaticStridesAttr());
        auto inputsOutputOperands2 = {static_cast<mlir::Value>(cmxToDDRClusterTilingCopyOp->getResult(0)),
                                      newSubView.getResult()};
        auto newClusterTilingOutType = newSubView.getResult().getType().cast<vpux::NDTypeInterface>();
        auto ddrToCMXClusterTilingCopyOp =
                rewriter.create<VPUIP::NCEClusterTilingOp>(appendLoc(inputClusterCopyOp->getLoc(), "_DDR_to_CMX_Copy"),
                                                           newClusterTilingOutType, inputsOutputOperands2, bodyBuilder);

        // remove old cluster tiling CopyOp
        rewriter.replaceOp(inputClusterCopyOp, ddrToCMXClusterTilingCopyOp->getResult(0));

        newConcatInputs.push_back(ddrToCMXClusterTilingCopyOp.getResults()[0]);
    }
}

mlir::Value MoveConcatViewWithClusteredCopyToCMX::rewriteViewLikeOps(mlir::Value input,
                                                                     ArrayRef<mlir::Operation*> viewLikeOps,
                                                                     VPUIP::DistributedBufferType origOutputBufferType,
                                                                     mlir::PatternRewriter& rewriter) const {
    auto ctx = rewriter.getContext();
    auto output = input;
    for (const auto& viewlikeOp : viewLikeOps) {
        if (auto reshapeOp = mlir::dyn_cast<VPUIP::GenericReshapeOp>(viewlikeOp)) {
            auto origType = reshapeOp.getOutput().getType().cast<NDTypeInterface>();
            const auto newType = getDuplicatedDistributedType(origType, origOutputBufferType, ctx);
            auto newReshapeOp = rewriter.create<VPUIP::GenericReshapeOp>(reshapeOp->getLoc(), newType, output);
            output = newReshapeOp.getOutput();
        } else if (auto shapeCastOp = mlir::dyn_cast<VPUIP::ShapeCastOp>(viewlikeOp)) {
            auto newShapeCastOp =
                    rewriter.create<VPUIP::ShapeCastOp>(shapeCastOp->getLoc(), output, shapeCastOp.getShape());
            output = newShapeCastOp.getResult();
        } else if (auto permuteCastOp = mlir::dyn_cast<VPUIP::PermuteCastOp>(viewlikeOp)) {
            auto origType = permuteCastOp.getResult().getType().cast<NDTypeInterface>();
            const auto newType = getDuplicatedDistributedType(origType, origOutputBufferType, ctx);
            auto newPermuteCastOp = rewriter.create<VPUIP::PermuteCastOp>(permuteCastOp->getLoc(), newType, output,
                                                                          permuteCastOp.getDstOrderAttr(),
                                                                          permuteCastOp.getMemPermAttr());
            output = newPermuteCastOp.getResult();
        } else {
            VPUX_THROW("Unsupported ViewLike Op");
        }
    }

    return output;
}

mlir::LogicalResult MoveConcatViewWithClusteredCopyToCMX::matchAndRewrite(VPUIP::ConcatViewOp concatViewOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    if (!concatViewOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    auto concatMemAlloc = VPUIP::getRootAlloc<mlir::memref::AllocOp>(concatViewOp.getOutputBuff());
    if (concatMemAlloc == nullptr) {
        _log.nest().trace("[{0}] Cannot rewrite because current concat '{1}' output isn't master buffer",
                          getDebugName(), concatViewOp->getLoc());
        return mlir::failure();
    }

    // Check inputs of ConcatView
    auto checkInputs = getValidConcatInputs(concatViewOp);
    if (mlir::failed(checkInputs)) {
        _log.nest().trace("[{0}] Invalid inputs for '{1}' at '{2}'", getDebugName(), concatViewOp->getName(),
                          concatViewOp->getLoc());
        return mlir::failure();
    }

    struct ConcatInputs concatInputs = checkInputs.value();

    // Check output of ConcatView
    auto checkOutputs = getValidConcatOutputs(concatViewOp);
    if (mlir::failed(checkOutputs)) {
        _log.nest().trace("[{0}] Invalid outputs for '{1}' at '{2}'", getDebugName(), concatViewOp->getName(),
                          concatViewOp->getLoc());
        return mlir::failure();
    }

    struct ConcatOutputs concatOutputs = checkOutputs.value();
    auto childClusterCopyOp = concatOutputs.outputClusterCopy;
    auto outputBuffer = childClusterCopyOp.getOutputBuffs()[0];
    const auto outputBufferType = outputBuffer.getType().dyn_cast<VPUIP::DistributedBufferType>();
    if (outputBufferType == nullptr) {
        _log.nest().trace("[{0}] ConcatView '{1}' at '{2}' user clustered copy buffer does not have distributedType",
                          getDebugName(), concatViewOp->getName(), concatViewOp->getLoc());
        return mlir::failure();
    }

    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), concatViewOp->getName(), concatViewOp->getLoc());

    // Create new subgraph to move ConcatView and viewlike ops to CMX
    auto ctx = rewriter.getContext();

    auto origConcatNDType = concatViewOp.getOutput().getType().cast<NDTypeInterface>();
    auto newConcatBufferType = getDuplicatedDistributedType(origConcatNDType, outputBufferType, ctx);
    // update buffer type so that new ConcatView can re-use this buffer on CMX
    outputBuffer.setType(newConcatBufferType);

    SmallVector<mlir::Value> newConcatInputs;
    rewriter.setInsertionPointAfter(outputBuffer.getDefiningOp());

    convertCopyInputAndStore(concatInputs.inputCopies, outputBuffer, newConcatInputs, rewriter);
    convertClusterTilingCopyInputAndStore(concatInputs.inputClusterCopies, outputBuffer, newConcatInputs, rewriter);
    auto newConcatOp = rewriter.create<VPUIP::ConcatViewOp>(concatViewOp->getLoc(), newConcatInputs, outputBuffer);

    auto subGraphOutput =
            rewriteViewLikeOps(newConcatOp.getOutput(), concatOutputs.viewLikeOps, outputBufferType, rewriter);

    // cast to original outputBufferType because alignment in distribution might be different
    auto distributedCastOp = rewriter.createOrFold<VPUIP::DistributedCastOp>(childClusterCopyOp->getLoc(),
                                                                             outputBufferType, subGraphOutput);

    rewriter.replaceOp(childClusterCopyOp, distributedCastOp);

    return mlir::success();
}

//
// OptimizeConcatViewCopiesPass
//

class OptimizeConcatViewCopiesPass final : public VPUIP::OptimizeConcatViewCopiesBase<OptimizeConcatViewCopiesPass> {
public:
    explicit OptimizeConcatViewCopiesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void OptimizeConcatViewCopiesPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AvoidConcatExtraChannel>(&ctx, _log);
    patterns.add<FuseConcatView>(&ctx, _log);
    patterns.add<RemoveDDRToDDRCopyAfterConcatView>(&ctx, _log);
    patterns.add<MoveConcatViewWithClusteredCopyToCMX>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeConcatViewCopiesPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createOptimizeConcatViewCopiesPass(Logger log) {
    return std::make_unique<OptimizeConcatViewCopiesPass>(log);
}
