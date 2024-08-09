//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/allocate_buffers.hpp"
#include "vpux/compiler/utils/dma.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"

using namespace vpux;

namespace {

//
// AvoidConcatExtraChannel
//

struct InputUpdateInfo {
    VPUIP::NCEClusterTilingOp tilingCopyOp;
    SmallVector<int64_t, 4> copyInOffsets;
    SmallVector<int64_t, 4> copyOutOffsets;
    SmallVector<int64_t, 4> copySizes;
};

class AvoidConcatExtraChannel : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    AvoidConcatExtraChannel(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp concatOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::LogicalResult checkConcatUsers(mlir::Value concatOutput, std::optional<int64_t>& patternOutChannelSize,
                                         std::optional<int64_t>& patternOutChannelOffset) const;
    mlir::LogicalResult checkConcatInputs(mlir::ValueRange concatInputs, mlir::Value concatOutput,
                                          const int64_t patternOutChannelSize, const int64_t patternOutChannelOffset,
                                          SmallVector<InputUpdateInfo>& inTilingCopiesInfo) const;

    mlir::Operation* createOutputBuffer(mlir::PatternRewriter& rewriter, VPUIP::NCEClusterTilingOp copyOp,
                                        int64_t channels) const;

    Logger _log;
};

// Check if all Concat users are Subview with same channels slice
// less than Concat channels (m > n)
//
//                      Concat (m output channels)
//                          |       ...       |
//  Subview (n output channels)     ...      Subview (n output channels)
//
mlir::LogicalResult AvoidConcatExtraChannel::checkConcatUsers(mlir::Value concatOutput,
                                                              std::optional<int64_t>& patternOutChannelSize,
                                                              std::optional<int64_t>& patternOutChannelOffset) const {
    auto subviews = concatOutput.getUsers();
    if (subviews.empty()) {
        return mlir::failure();
    }

    const auto concatOutChannelSize = getShape(concatOutput)[Dims4D::Act::C];
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

        if (patternOutChannelOffset.has_value() && patternOutChannelOffset.value() != offsets[Dims4D::Act::C.ind()]) {
            return mlir::failure();
        }

        if (patternOutChannelSize.has_value() && patternOutChannelSize.value() != sizes[Dims4D::Act::C.ind()]) {
            return mlir::failure();
        }

        if (concatOutChannelSize <= sizes[Dims4D::Act::C.ind()]) {
            return mlir::failure();
        }

        patternOutChannelSize = sizes[Dims4D::Act::C.ind()];
        patternOutChannelOffset = offsets[Dims4D::Act::C.ind()];
    }

    return mlir::success();
}

// Check if all Concat inputs copy NCE result with more channels
// than Subview after Concat
//
// Scenario 1: Concat joins its inputs not by channel dimension (m > n)
//
//                 Input0                      Input1
//                    |                           |
//        TilingCopy (m channels)     TilingCopy (m channels)
//            |               |          |                 |
//    Subview (m)          Concat (m channels)          Subview (m)
//                                 |
//                        Subview (n channels)
//
// Scenario 2: Concat joins its inputs by channel dimension (p > m && p > n && p > q)
//
//                 Input0                      Input1
//                    |                           |
//        TilingCopy (m channels)      TilingCopy (n channels)
//            |               |          |                  |
//    Subview (m)          Concat (p channels)           Subview (n)
//                                  |
//                         Subview (q channels)
//
mlir::LogicalResult AvoidConcatExtraChannel::checkConcatInputs(mlir::ValueRange concatInputs, mlir::Value concatOutput,
                                                               const int64_t patternOutChannelSize,
                                                               const int64_t patternOutChannelOffset,
                                                               SmallVector<InputUpdateInfo>& inTilingCopiesInfo) const {
    if (concatInputs.empty() || concatOutput == nullptr) {
        return mlir::failure();
    }

    auto getConcatDims = [](ShapeRef inShape, ShapeRef outShape) {
        VPUX_THROW_UNLESS(inShape.size() == outShape.size(), "Got unexpect input and output shape");
        SmallVector<Dim> concatDims;
        auto ioShapes = zip(inShape, outShape);
        for (const auto& ioShape : ioShapes | indexed) {
            const auto inSize = std::get<0>(ioShape.value());
            const auto outSize = std::get<1>(ioShape.value());
            if (inSize != outSize) {
                concatDims.push_back(Dim(ioShape.index()));
            }
        }
        return concatDims;
    };

    const auto concatOutShape = getShape(concatOutput);
    const auto concatChannels = concatOutShape[Dims4D::Act::C];
    for (auto input : concatInputs) {
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

        if (VPUIP::getRootAlloc<mlir::memref::AllocOp>(subview.getSource()) == nullptr) {
            return mlir::failure();
        }

        if (subview.getStaticStrides().has_value()) {
            return mlir::failure();
        }

        auto offsets = parseIntArrayAttr<int64_t>(subview.getStaticOffsetsAttr());
        auto sizes = parseIntArrayAttr<int64_t>(subview.getStaticSizesAttr());

        auto concatDims = getConcatDims(Shape(sizes), concatOutShape);
        if (concatDims.size() != 1) {
            return mlir::failure();
        }

        SmallVector<int64_t, 4> copyInOffsets(4, 0);
        SmallVector<int64_t, 4> copyOutOffsets(offsets.begin(), offsets.end());
        SmallVector<int64_t, 4> copySizes(sizes.begin(), sizes.end());
        const auto channelIdx = Dims4D::Act::C.ind();
        if (concatDims.front() == Dims4D::Act::C) {
            const auto currentInputChannelBegin = offsets[channelIdx];
            const auto currentInputChannelEnd = currentInputChannelBegin + sizes[channelIdx];
            const auto patternChannelBegin = patternOutChannelOffset;
            const auto patternChannelEnd = patternChannelBegin + patternOutChannelSize;

            bool isPatternBeginInside =
                    patternChannelBegin >= currentInputChannelBegin && patternChannelBegin <= currentInputChannelEnd;
            bool isPatternEndInside =
                    patternChannelEnd >= currentInputChannelBegin && patternChannelEnd <= currentInputChannelEnd;

            copyOutOffsets[channelIdx] = currentInputChannelBegin - patternChannelBegin;
            // ConcatView Across Channels:
            // |   Input_0   |   Input_1   |   Input_2   |   ...   |   Input_n   |
            // When the slice starts inside Input_0, it's legal scenario:
            //    | Input_0' |   Input_1   |   Input_2   |   ...   |   Input_n   |
            // When the slice starts inside an input other than Input_0, it is illegal scenario:
            //                   | Input_1'|   Input_2   |   ...   |   Input_n   |
            // When both the pattern start and end are inside an input, it is illegal scenario:
            //   |Input_0'|
            // For these two illegal cases, some inputs are unnecessary and can be removed
            // Although unlikely to appear in real models, it's better to have checks in place
            if (isPatternBeginInside) {
                if (currentInputChannelBegin != 0 || isPatternEndInside) {
                    return mlir::failure();
                }
                copyInOffsets[channelIdx] = patternChannelBegin;
                copyOutOffsets[channelIdx] = 0;
                copySizes[channelIdx] = sizes[channelIdx] - patternChannelBegin;
            }

            // Similar logic to isPatternBeginInside scenario
            // The slice must start in the first input and end in the last input
            if (isPatternEndInside) {
                if (currentInputChannelEnd != concatChannels || isPatternBeginInside) {
                    return mlir::failure();
                }
                copyInOffsets[channelIdx] = 0;
                copySizes[channelIdx] = patternChannelEnd - currentInputChannelBegin;
            }
        } else {
            copyInOffsets[channelIdx] = patternOutChannelOffset;
            copySizes[channelIdx] = patternOutChannelSize;
        }

        auto copyOpInput = tilingCopy.getInputs()[0];
        if (auto distributedType = copyOpInput.getType().dyn_cast<VPUIP::DistributedBufferType>()) {
            const auto tileIndex = VPUIP::getTilingDimIndex(distributedType);
            if (tileIndex.has_value()) {
                auto tileIndexVal = tileIndex.value();
                if (!VPUIP::isChannelOffsetsAndTileDimCompatibleWithClusterCopy(to_small_vector(copyInOffsets),
                                                                                tileIndexVal, distributedType)) {
                    return mlir::failure();
                }
            }
        }

        inTilingCopiesInfo.push_back(
                InputUpdateInfo{tilingCopy, std::move(copyInOffsets), std::move(copyOutOffsets), std::move(copySizes)});
    }

    return mlir::success();
}

mlir::Operation* AvoidConcatExtraChannel::createOutputBuffer(mlir::PatternRewriter& rewriter,
                                                             VPUIP::NCEClusterTilingOp copyOp, int64_t channels) const {
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

// Scenario 1: Concat joins its inputs not by channel dimension (m > n)
//
//                 Input0                      Input1
//                    |                           |
//        TilingCopy (m channels)     TilingCopy (m channels)
//            |               |          |                 |
//    Subview (m)          Concat (m channels)          Subview (m)
//                                 |
//                        Subview (n channels)
//
// is converted to pattern
//
//        Input0 (m channels)             Input1 (m channels)
//                  |                            |
//        Subview (n channels)           Subview (n channels)
//                  |                            |
//        TilingCopy (n channels)     TilingCopy (n channels)
//           |               |            |            |
//    Subview (n)         Concat (n channels)        Subview (n)
//
// Scenario 2: Concat joins its inputs by channel dimension (p > m && p > n && p > q)
//
//                 Input0                      Input1
//                    |                           |
//        TilingCopy (m channels)      TilingCopy (n channels)
//            |               |          |                  |
//    Subview (m)          Concat (p channels)           Subview (n)
//                                  |
//                         Subview (q channels)
//
// is converted to pattern
//
//        Input0 (m channels)     Input1 (n channels)
//                |                         |
//                |               Subview (n - (p - q) channels)
//                |                         |
//    TilingCopy (m channels)     TilingCopy (n - (p - q) channels)
//           |           |           |           |
//    Subview (m)      Concat (q channels)    Subview (n - (p - q))
//
mlir::LogicalResult AvoidConcatExtraChannel::matchAndRewrite(VPUIP::ConcatViewOp concatOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("Got VPUIP.ConcatViewOp at '{0}'", concatOp->getLoc());
    auto nestedLogger = _log.nest();

    auto concatOutput = concatOp.getOutput();
    if (getShape(concatOutput).size() != 4) {
        nestedLogger.trace("Cannot optimize because of shape rank not being 4");
        return mlir::failure();
    }

    std::optional<int64_t> patternOutChannelSize = std::nullopt;
    std::optional<int64_t> patternOutChannelOffset = std::nullopt;
    if (checkConcatUsers(concatOutput, patternOutChannelSize, patternOutChannelOffset).failed()) {
        nestedLogger.trace("Cannot optimize because of users requirements");
        return mlir::failure();
    }

    auto concatInputs = concatOp.getInputs();
    SmallVector<InputUpdateInfo> inTilingCopiesInfo;
    inTilingCopiesInfo.reserve(concatInputs.size());
    if (checkConcatInputs(concatInputs, concatOutput, patternOutChannelSize.value(), patternOutChannelOffset.value(),
                          inTilingCopiesInfo)
                .failed()) {
        nestedLogger.trace("Cannot optimize because of input requirements");
        return mlir::failure();
    }

    auto* outputBuffer =
            createOutputBuffer(rewriter, inTilingCopiesInfo.front().tilingCopyOp, patternOutChannelSize.value());
    if (outputBuffer == nullptr) {
        nestedLogger.trace("Cannot allocate new output buffer");
        return mlir::failure();
    }

    SmallVector<mlir::Value> newConcatInputs;
    newConcatInputs.reserve(concatInputs.size());
    for (auto inTilingCopyInfo : inTilingCopiesInfo) {
        auto copyOp = inTilingCopyInfo.tilingCopyOp;
        auto copyOpInput = copyOp.getInputs()[0];
        auto copyOpOutput = copyOp.getOutputs()[0];

        auto subview = copyOpOutput.getDefiningOp<VPUIP::SubViewOp>();

        auto newCopyInSubview = copyOpInput;
        if (Shape(inTilingCopyInfo.copySizes) != getShape(copyOpInput)) {
            newCopyInSubview = rewriter.create<VPUIP::SubViewOp>(
                    subview.getLoc(), copyOpInput,
                    getIntArrayAttr(subview.getContext(), inTilingCopyInfo.copyInOffsets),
                    getIntArrayAttr(subview.getContext(), inTilingCopyInfo.copySizes));
        }

        auto newCopyOutSubview = rewriter.create<VPUIP::SubViewOp>(
                subview.getLoc(), outputBuffer->getResult(0),
                getIntArrayAttr(subview.getContext(), inTilingCopyInfo.copyOutOffsets),
                getIntArrayAttr(subview.getContext(), inTilingCopyInfo.copySizes));

        const auto copyOutBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                            mlir::ValueRange newOperands) {
            builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
        };

        SmallVector<mlir::Value> inputsOutputOperands = {newCopyInSubview, newCopyOutSubview};
        auto newTilingCopy = rewriter.create<VPUIP::NCEClusterTilingOp>(copyOp->getLoc(), newCopyOutSubview.getType(),
                                                                        inputsOutputOperands, copyOutBodyBuilder);

        newConcatInputs.push_back(newTilingCopy.getResults()[0]);
    }

    auto newOp =
            rewriter.replaceOpWithNewOp<VPUIP::ConcatViewOp>(concatOp, newConcatInputs, outputBuffer->getResult(0));

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

    for (auto inTilingCopyInfo : inTilingCopiesInfo) {
        rewriter.eraseOp(inTilingCopyInfo.tilingCopyOp);
    }

    nestedLogger.trace("Successfully Avoid Concat Extra Channel");
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
        return VPU::isDuplicated(distribution);
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
        return VPU::isDuplicated(distribution);
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
// OptimizeConcatSubviewPattern
//

class OptimizeConcatSubviewPattern : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
public:
    OptimizeConcatSubviewPattern(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp concatOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

/*
Optimize subgraph like below, note that for copy0 and copy2, copy1 and copy3, they should have same size and offsets
  Input0      Input1
    |           |
   Copy0      Copy1
     \        /                 |         |
     ConcatView       =>      Input0    Input1
     /        \                 |         |
  Subview0   Subview1
    |           |
   Copy2      Copy3

*/
mlir::LogicalResult OptimizeConcatSubviewPattern::matchAndRewrite(VPUIP::ConcatViewOp concatOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    auto nestedLogger = _log.nest();

    auto concatOutput = concatOp.getOutput();
    if (getShape(concatOutput).size() != 4) {
        nestedLogger.trace("Cannot optimize because of shape rank not being 4");
        return mlir::failure();
    }

    SmallVector<VPUIP::SubViewOp> inputSubViews;
    SmallVector<VPUIP::SubViewOp> outputSubViews;
    SmallVector<VPUIP::NCEClusterTilingOp> inputTilingCopies;
    SmallVector<VPUIP::NCEClusterTilingOp> outputTilingCopies;

    // check input
    for (auto input : concatOp.getInputs()) {
        auto tilingCopy = input.getDefiningOp<VPUIP::NCEClusterTilingOp>();
        if (tilingCopy == nullptr || !tilingCopy->hasOneUse() ||
            tilingCopy.getInnerTaskOpOfType<VPUIP::CopyOp>() == nullptr) {
            return mlir::failure();
        }

        auto copyOpOutput = tilingCopy.getOutputs()[0];
        auto subview = copyOpOutput.getDefiningOp<VPUIP::SubViewOp>();
        if (subview == nullptr) {
            return mlir::failure();
        }

        if (VPUIP::getRootAlloc<mlir::memref::AllocOp>(subview.getSource()) == nullptr) {
            return mlir::failure();
        }

        inputSubViews.push_back(subview);
        inputTilingCopies.push_back(tilingCopy);
    }

    // check output
    for (auto user : concatOp->getUsers()) {
        auto subview = mlir::dyn_cast_or_null<VPUIP::SubViewOp>(user);
        if (subview == nullptr || !subview->hasOneUse()) {
            return mlir::failure();
        }

        auto tilingCopy = mlir::dyn_cast_or_null<VPUIP::NCEClusterTilingOp>(*(subview->getUsers().begin()));
        if (tilingCopy == nullptr || tilingCopy.getInnerTaskOpOfType<VPUIP::CopyOp>() == nullptr) {
            return mlir::failure();
        }

        outputSubViews.push_back(subview);
        outputTilingCopies.push_back(tilingCopy);
    }

    if (inputSubViews.empty() || (outputSubViews.empty())) {
        return mlir::failure();
    }

    auto isSameAttrSubview = [](VPUIP::SubViewOp inSubview, VPUIP::SubViewOp outSubview) {
        return inSubview.getStaticOffsetsAttr() == outSubview.getStaticOffsetsAttr() &&
               inSubview.getStaticSizesAttr() == outSubview.getStaticSizesAttr() &&
               inSubview.getStaticStridesAttr() == outSubview.getStaticStridesAttr();
    };
    auto isSameDistributedTypeCopy = [](VPUIP::NCEClusterTilingOp inCopy, VPUIP::NCEClusterTilingOp outCopy) {
        auto inputValue = inCopy.getInputs()[0];
        auto inputDistributedType = inputValue.getType().dyn_cast<VPUIP::DistributedBufferType>();
        if (inputDistributedType == nullptr) {
            return false;
        }

        auto outputValue = outCopy->getResult(0);
        auto outputDistributedType = outputValue.getType().dyn_cast<VPUIP::DistributedBufferType>();
        if (outputDistributedType == nullptr) {
            return false;
        }
        return mlir::succeeded(VPU::isDistributedCastCompatible(inputDistributedType, outputDistributedType));
    };

    mlir::DenseMap<int64_t, int64_t> outIndexToInIndex;
    auto inRange = irange(inputSubViews.size());
    for (auto outIdx : irange(outputSubViews.size())) {
        auto iter = llvm::find_if(inRange, [&](auto inIdx) {
            return isSameAttrSubview(inputSubViews[inIdx], outputSubViews[outIdx]) &&
                   isSameDistributedTypeCopy(inputTilingCopies[inIdx], outputTilingCopies[outIdx]);
        });
        if (iter == inRange.end()) {
            return mlir::failure();
        }
        auto inIdx = std::distance(inRange.begin(), iter);
        outIndexToInIndex[outIdx] = inIdx;
    }

    nestedLogger.trace("optimize concat->subview pattern at {0}", concatOp->getLoc());

    for (const auto& item : outIndexToInIndex) {
        const auto& outIdx = item.first;
        const auto& inIdx = item.second;
        auto& outputCopy = outputTilingCopies[outIdx];
        auto& inputCopy = inputTilingCopies[inIdx];
        auto newOutput = rewriter.createOrFold<VPUIP::DistributedCastOp>(
                outputCopy->getLoc(), outputCopy.getResult(0).getType(), inputCopy.getInputs()[0]);

        rewriter.replaceAllUsesWith(outputCopy.getResult(0), newOutput);
        rewriter.eraseOp(outputCopy);
        rewriter.eraseOp(outputSubViews[outIdx]);
    }

    rewriter.eraseOp(concatOp);

    for (auto ind : irange(inputTilingCopies.size())) {
        rewriter.eraseOp(inputTilingCopies[ind]);
        rewriter.eraseOp(inputSubViews[ind]);
    }

    return mlir::success();
}

template <class OpType>
OpType getSingleUserOfType(mlir::Operation* op) {
    return op->hasOneUse() ? mlir::dyn_cast<OpType>(*op->getUsers().begin()) : nullptr;
}

/*
    LeftBranch: [View]BlockArg(DDR) ------------\                                              /  SubView0 -> TiledCopy
                                                 \                                            /   SubView1 -> TiledCopy
                                                 | -> Concat -> GenericReshape -> PermCast -> |   SubView2 -> TiledCopy
                                                 /                                            \   SubView3 -> TiledCopy
    RightBranch:  DistrBuf(CMX) -> TiledCopy  --/                                              \  SubViewN -> TiledCopy

    To

    LeftBranch  -> GenericReshape -> PermCast \
    RightBranch -> GenericReshape -> PermCast |
                                              +-> Subview0Left  \
                                              |                 |-> Concat -> NCEClusterTiling
                                              +-> Subview0Right /
                                              ...
                                              +-> SubviewNLeft  \
                                              |                 |-> Concat -> NCEClusterTiling
                                              +-> SubviewNRight /

    In case of large DDR input, what is common for KV cache models, original pattern creates large DDR->DDR,
    which scheduled in the beginning of the inference and blocks prefetch/execution. New pattern is more friendly
    and can be distributed across schedule.
    This pattern has specific requirements for GenericReshape, see checkConcatReshapeCompatibility

    Concat(Left[1, 32, 128, 1023], Right[1, 32, 128, 1]) -> Reshape[32 * 128, 1024] -> PermCast -> Views
    to
    Reshape [32 * 128, 1]    -> PermCast -> View -\
                                                  |-> Concat
    Reshape [32 * 128, 1023] -> PermCast -> View -/
*/
class SplitUnbalancedDDRConcatBase : public mlir::OpRewritePattern<VPUIP::ConcatViewOp> {
protected:
    // Describes parameters of original concat.
    struct PatternParamsInfo {
        int64_t leftConcatInputSize;  // Dim size of left buffer on concat axis.
        int64_t leftInputSize;   // If left buffer was viewed, DimSize of input, otherwise equals to leftConcatInputSize
        int64_t leftViewOffset;  // Offset from begging if was viewed
        int64_t rightInputSize;
        Dim origConcatDim;
        Dim newConcatDim;  // Concat Dim after GReshape
        VPUIP::PermuteCastOp castOp;
    };

public:
    SplitUnbalancedDDRConcatBase(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUIP::ConcatViewOp>(ctx), _log(log) {
    }

private:
    // Suffix of child rewriter
    virtual StringRef getRewriterSuffix() const = 0;
    // Check compatibility of concat in CMX. Since consumers are segmented, we must ensure that we can do split.
    // If view/concat axis is the same with tiling axis, we can't do concatenation in same buffer.
    // To avoid it we must do concat in temporary buffer and then distribute
    virtual bool isSplitSupported(Dim newConcatDim, int64_t tilingDim) const = 0;

    // Input of the right side and copy operation
    virtual std::pair<mlir::Value, mlir::Operation*> getRightBranchInput(VPUIP::ConcatViewOp concatOp) const = 0;

    virtual mlir::Value prepareRightBranch(mlir::PatternRewriter& rewriter, mlir::Value rightBranchInput,
                                           mlir::Operation* rightBranchCopy, VPUIP::PermuteCastOp permuteCastOp,
                                           mlir::Location loc) const = 0;

    virtual mlir::Value createNewConcatBuffer(mlir::PatternRewriter& rewriter, VPUIP::SubViewOp origView,
                                              VPUIP::NCEClusterTilingOp clusterCopy,
                                              mlir::Location bufferLoc) const = 0;

    virtual void rewriteSubview(mlir::PatternRewriter& rewriter, VPUIP::ConcatViewOp origConcatOp,
                                VPUIP::SubViewOp subViewOp, VPUIP::NCEClusterTilingOp clusterCopy,
                                mlir::Value newLeftBranch, mlir::Value newRightBranch, const PatternParamsInfo& params,
                                size_t index) const = 0;

protected:
    // Propage reshape and cast through concat to split large DDR->DDR DMAs on sequence of small DDR->CMX, which can be
    // interleaved with NCE tasks
    mlir::Value propagateReshapeCast(mlir::PatternRewriter& rewriter, mlir::Value branchInput,
                                     VPUIP::PermuteCastOp permuteCastOp, mlir::Location loc,
                                     StringRef locSuffix) const {
        auto inputType = branchInput.getType().cast<vpux::NDTypeInterface>();
        auto origShape = inputType.getShape();
        Shape newShape = origShape.toValues();
        newShape[Dim(0)] = origShape[Dim(1)] * origShape[Dim(2)];
        newShape[Dim(1)] = origShape[Dim(3)];
        newShape[Dim(2)] = 1;
        newShape[Dim(3)] = 1;
        auto afterReshapeType = inputType.changeShape(newShape);
        auto newReshapeOp = rewriter.createOrFold<VPUIP::GenericReshapeOp>(appendLoc(loc, "greshape_{0}", locSuffix),
                                                                           afterReshapeType, branchInput);

        auto permCastDimsOrder = permuteCastOp->getResultTypes()[0].cast<vpux::NDTypeInterface>().getDimsOrder();
        auto afterPermCastType = afterReshapeType.changeDimsOrder(permCastDimsOrder);
        return rewriter.create<VPUIP::PermuteCastOp>(appendLoc(loc, "permcast_{0}", locSuffix), afterPermCastType,
                                                     newReshapeOp, permuteCastOp.getDstOrderAttr(),
                                                     permuteCastOp.getMemPermAttr());
    }

    mlir::Value createNewCopyBranch(mlir::PatternRewriter& rewriter, mlir::Value src, mlir::Value dst,
                                    ShapeRef copyShape, ShapeRef srcOffset, ShapeRef dstOffset, mlir::Location baseLoc,
                                    StringRef locSuffix, size_t opId) const {
        mlir::Value srcView = rewriter.createOrFold<VPUIP::SubViewOp>(
                appendLoc(baseLoc, "{0}_src_view_{1}", locSuffix, opId), src, srcOffset, copyShape);
        mlir::Value dstView = rewriter.createOrFold<VPUIP::SubViewOp>(
                appendLoc(baseLoc, "{0}_dst_view_{1}", locSuffix, opId), dst, dstOffset, copyShape);

        auto copyLoc = appendLoc(baseLoc, "{0}_copy_{1}", locSuffix, opId);
        if (dst.getDefiningOp<VPURT::AllocDistributed>() != nullptr) {
            const auto copyOutBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                                mlir::ValueRange newOperands) {
                builder.create<VPUIP::CopyOp>(loc, newOperands[0], newOperands[1]);
            };
            SmallVector<mlir::Value> inputsOutputOperands{srcView, dstView};
            return rewriter
                    .create<VPUIP::NCEClusterTilingOp>(copyLoc, dstView.getType(), inputsOutputOperands,
                                                       copyOutBodyBuilder)
                    ->getResult(0);
        } else {
            return rewriter.create<VPUIP::CopyOp>(copyLoc, srcView, dstView);
        }
    }

private:
    bool checkConcatReshapeCompatibility(VPUIP::ConcatViewOp concatOp, VPUIP::GenericReshapeOp genReshapeOp,
                                         vpux::Logger log) const {
        auto genReshapeType = vpux::getBufferType(genReshapeOp.getOutput());
        auto concatType = vpux::getBufferType(concatOp.getOutput());

        if (genReshapeType.getRank() != 4 || concatType.getRank() != 4) {
            log.trace("Only 4D tensors are supported");
            return false;
        }

        auto reshapeShape = genReshapeType.getShape();
        auto concatShape = concatType.getShape();
        if (concatShape[Dim(0)] != 1) {
            log.trace("Only Batch size 1 is supported");
            return false;
        }

        // [1, A, B, C] -> [A*B, C, 1, 1]
        bool compatible = reshapeShape[Dim(0)] == concatShape[Dim(1)] * concatShape[Dim(2)] &&
                          reshapeShape[Dim(1)] == concatShape[Dim(3)] &&
                          genReshapeType.getNumElements() == concatType.getNumElements();
        if (!compatible) {
            log.trace("Concat->Reshape shapes are not compatible: {0} vs {1}", concatType, reshapeShape);
            return false;
        }
        return true;
    }

    // Input of the left side, must be block argument
    std::pair<mlir::Value, VPUIP::SubViewOp> getLeftBranchInput(VPUIP::ConcatViewOp concatOp) const {
        const size_t LEFT_INPUT_ID = 0;  // Left must be always first to preserve concat order
        auto inputCopy = concatOp.getInputs()[LEFT_INPUT_ID];
        if (auto copyOp = inputCopy.getDefiningOp<VPUIP::CopyOp>()) {
            auto input = copyOp.getInput();
            if (mlir::isa<mlir::BlockArgument>(input)) {
                return {input, nullptr};
            }
            if (auto viewOp = input.getDefiningOp<VPUIP::SubViewOp>()) {
                auto viewInput = viewOp.getSource();
                auto validInputView = viewOp->hasOneUse() && mlir::isa<mlir::BlockArgument>(viewInput);
                if (validInputView) {
                    return {viewInput, viewOp};
                }
            }
        }
        return {nullptr, nullptr};
    }

    mlir::Value prepareLeftBranch(mlir::PatternRewriter& rewriter, mlir::Value leftBranchInput,
                                  VPUIP::PermuteCastOp permuteCastOp, mlir::Location loc) const {
        return propagateReshapeCast(rewriter, leftBranchInput, permuteCastOp, loc, "left");
    }

    std::optional<int64_t> getTilingAxis(mlir::Type type) const {
        auto outDistributedType = type.dyn_cast<VPUIP::DistributedBufferType>();
        if (outDistributedType == nullptr) {
            return std::nullopt;
        }

        const auto distAttr = outDistributedType.getDistribution();
        const auto distMode = distAttr.getMode().getValue();
        if (distMode != VPU::DistributionMode::SEGMENTED) {
            return std::nullopt;
        }
        const auto numTiles = distAttr.getNumTiles();
        const auto tilingScheme = parseIntArrayAttr<int64_t>(numTiles);
        return VPU::getDistributedTilingAxis(tilingScheme);
    };

public:
    mlir::LogicalResult matchAndRewrite(VPUIP::ConcatViewOp concatOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("SplitUnbalancedDDRConcat{0}: Got Concat at '{1}'", getRewriterSuffix(), concatOp.getLoc());
        auto nestedLog = _log.nest();

        if (concatOp.getInputs().size() != 2) {
            nestedLog.trace("Only 2 inputs supported");
            return mlir::failure();
        }

        // Output type is same as type of all inputs
        auto isDdrConcat = vpux::getBufferType(concatOp).getMemoryKind() == VPU::MemoryKind::DDR;
        if (!isDdrConcat) {
            nestedLog.trace("All inputs must be DDR copies");
            return mlir::failure();
        }

        auto genReshapeOp = getSingleUserOfType<VPUIP::GenericReshapeOp>(concatOp);
        if (genReshapeOp == nullptr) {
            nestedLog.trace("ConcatOp must followed by only one GenericReshape");
            return mlir::failure();
        }
        if (!checkConcatReshapeCompatibility(concatOp, genReshapeOp, nestedLog)) {
            return mlir::failure();
        }

        auto permuteCastOp = getSingleUserOfType<VPUIP::PermuteCastOp>(genReshapeOp);
        if (permuteCastOp == nullptr) {
            nestedLog.trace("GenericReshape must followed by only one PermuteCast");
            return mlir::failure();
        }

        const auto concatAxes =
                vpux::IE::getDiffInOutSizeDims(getShape(concatOp.getOperands()[0]), getShape(concatOp.getResult()));
        if (concatAxes.size() != 1) {
            nestedLog.trace("Cannot extract concat axis");
            return mlir::failure();
        }
        auto origConcatDim = concatAxes.front();
        Dim newConcatDim(0);
        // 0 - invalid, 1-2 -> 0, 3->1
        if (origConcatDim == Dim(0)) {
            nestedLog.trace("Unsupported orig concat dim {0}", origConcatDim);
            return mlir::failure();
        }
        if (origConcatDim == Dim(3)) {
            newConcatDim = Dim(1);
        }
        // GenericReshape collapses 1,2 axis into 0, and 3 to 1, see checkConcatReshapeCompatibility
        nestedLog.trace("Concat axis transformation: {0} -> {1}", origConcatDim, newConcatDim);

        auto [leftBranchInput, leftBranchViewOp] = getLeftBranchInput(concatOp);
        if (leftBranchInput == nullptr) {
            nestedLog.trace("Can't get left branch input");
            return mlir::failure();
        }
        SmallVector<int64_t> leftViewOffsets;
        int64_t leftSizeOnConcatDim = getShape(leftBranchInput)[origConcatDim];
        int64_t leftViewOffsetOnConcatDim = 0;
        if (leftBranchViewOp != nullptr) {
            leftViewOffsets = parseIntArrayAttr<int64_t>(leftBranchViewOp.getStaticOffsets());
            auto viewOffsetOnConcatDim = leftViewOffsets[origConcatDim.ind()];
            if (viewOffsetOnConcatDim != 1) {
                nestedLog.trace("Only off by one offset is supported");
                return mlir::failure();
            }
            leftSizeOnConcatDim = getShape(leftBranchViewOp->getResult(0))[origConcatDim];
            leftViewOffsetOnConcatDim = viewOffsetOnConcatDim;
        }

        auto [rightBranchInput, rightBranchCopyOp] = getRightBranchInput(concatOp);
        if (rightBranchInput == nullptr || rightBranchCopyOp == nullptr) {
            nestedLog.trace("Can't get right branch input");
            return mlir::failure();
        }
        VPUX_THROW_WHEN(leftBranchInput == rightBranchInput, "Branches must have different inputs");

        SmallVector<VPUIP::SubViewOp> views;
        SmallVector<VPUIP::NCEClusterTilingOp> clusteredCopies;
        for (auto user : permuteCastOp->getUsers()) {
            if (auto viewOp = mlir::dyn_cast<VPUIP::SubViewOp>(user)) {
                if (!viewOp->hasOneUse()) {
                    nestedLog.trace("ViewOp at '{0}' must have only one user", viewOp->getLoc());
                    return mlir::failure();
                }
                views.push_back(viewOp);

                auto clusterOp = getSingleUserOfType<VPUIP::NCEClusterTilingOp>(viewOp);
                if (clusterOp == nullptr || clusterOp.getInnerTaskOpOfType<VPUIP::CopyOp>() == nullptr) {
                    nestedLog.trace("View at '{0}' user is not a Clustered Copy", viewOp->getLoc());
                    return mlir::failure();
                }
                clusteredCopies.push_back(clusterOp);
            } else {
                nestedLog.trace("All users must be View operations");
                return mlir::failure();
            }
        }
        if (views.empty()) {
            nestedLog.trace("Cannot find any SubView->NCEClusterTiling[Copy] consumers");
            return mlir::failure();
        }

        if (clusteredCopies.size() < 1) {
            nestedLog.trace("Expected at least 2 users after concat");
            return mlir::failure();
        }

        auto maybeTilingAxis = getTilingAxis(clusteredCopies.front().getResultTypes()[0]);
        if (!maybeTilingAxis.has_value()) {
            nestedLog.trace("Only SEGMENTED distribution is supported for consumers");
            return mlir::failure();
        }

        auto tilingAxis = maybeTilingAxis.value();
        if (tilingAxis != 0) {
            nestedLog.trace("Only tiling on major dim is supported");
            return mlir::failure();
        }

        auto allTilingAxisAreSame = llvm::all_of(clusteredCopies, [&](VPUIP::NCEClusterTilingOp tiledCopy) {
            auto currentTilingAxis = getTilingAxis(tiledCopy.getResultTypes()[0]);
            return currentTilingAxis.has_value() && currentTilingAxis.value() == tilingAxis;
        });
        if (!allTilingAxisAreSame) {
            nestedLog.trace("Concat users have different distribution axis");
            return mlir::failure();
        }

        nestedLog.trace("Found {0} copies to split. newConcatDim: {1}, segmentationDim: d{2}", views.size(),
                        newConcatDim, tilingAxis);
        if (!isSplitSupported(newConcatDim, tilingAxis)) {
            nestedLog.trace("Not supported combination of tiling/concat");
            return mlir::failure();
        }
        PatternParamsInfo params{leftSizeOnConcatDim,
                                 getShape(leftBranchInput)[origConcatDim],
                                 leftViewOffsetOnConcatDim,
                                 getShape(rightBranchInput)[origConcatDim],
                                 origConcatDim,
                                 newConcatDim,
                                 permuteCastOp};

        mlir::Value newLeftBranch = prepareLeftBranch(rewriter, leftBranchInput, permuteCastOp, concatOp->getLoc());
        mlir::Value newRightBranch =
                prepareRightBranch(rewriter, rightBranchInput, rightBranchCopyOp, permuteCastOp, concatOp->getLoc());

        for (size_t i = 0; i < clusteredCopies.size(); ++i) {
            VPUIP::NCEClusterTilingOp clusterCopy = clusteredCopies[i];
            VPUIP::SubViewOp subViewOp = views[i];

            rewriter.setInsertionPoint(clusterCopy);
            rewriteSubview(rewriter, concatOp, subViewOp, clusterCopy, newLeftBranch, newRightBranch, params, i);
        }

        SmallVector<mlir::Value> concatInputs(concatOp.getInputs());
        rewriter.eraseOp(permuteCastOp);
        rewriter.eraseOp(genReshapeOp);
        rewriter.eraseOp(concatOp);
        for (mlir::Value val : concatInputs) {
            if (auto defOp = val.getDefiningOp()) {
                if (defOp->use_empty()) {
                    rewriter.eraseOp(defOp);
                }
            }
        }
        nestedLog.trace("Done");
        _log.unnest();
        return mlir::success();
    }

private:
    Logger _log;
};

/*
    Concat(Left[1, 32, 1023, 128], Right[1, 32, 1, 128]) -> Reshape[32 * 1024, 128] -> PermCast -> Views
    to
    Reshape [32 * 1023, 128] -> PermCast -> View -\
                                                  |-> Concat -> TiledCopy to CMX
    FlatView -> Reshape [1, 128] -> PermCast     -/
*/
class SplitUnbalancedDDRConcatOnOtherAxis : public SplitUnbalancedDDRConcatBase {
public:
    using SplitUnbalancedDDRConcatBase::SplitUnbalancedDDRConcatBase;

private:
    StringRef getRewriterSuffix() const override {
        return "OnOtherAxis";
    }

    bool isSplitSupported(Dim newConcatDim, int64_t tilingDim) const override {
        return newConcatDim.ind() != tilingDim;
    }

    std::pair<mlir::Value, mlir::Operation*> getRightBranchInput(VPUIP::ConcatViewOp concatOp) const override {
        const size_t RIGHT_INPUT_ID = 1;  // Right must be always second to preserve concat order
        auto inputCopy = concatOp.getInputs()[RIGHT_INPUT_ID];
        if (auto copyOp = inputCopy.getDefiningOp<VPUIP::CopyOp>()) {
            if (!mlir::isa<mlir::BlockArgument>(copyOp.getInput())) {
                return {copyOp.getInput(), copyOp};
            }
        }
        return {nullptr, nullptr};
    }

    mlir::Value prepareRightBranch(mlir::PatternRewriter& rewriter, mlir::Value rightBranchInput, mlir::Operation*,
                                   VPUIP::PermuteCastOp permuteCastOp, mlir::Location loc) const override {
        return propagateReshapeCast(rewriter, rightBranchInput, permuteCastOp, loc, "right");
    }

    // Buffer remains the same
    mlir::Value createNewConcatBuffer(mlir::PatternRewriter& rewriter, VPUIP::SubViewOp,
                                      VPUIP::NCEClusterTilingOp clusterCopy, mlir::Location bufferLoc) const override {
        auto dstBufferType = clusterCopy.getOutputBuffs()[0].getType();
        return rewriter.create<VPURT::AllocDistributed>(bufferLoc, dstBufferType, nullptr, nullptr);
    }

    void rewriteSubview(mlir::PatternRewriter& rewriter, VPUIP::ConcatViewOp origConcatOp, VPUIP::SubViewOp subViewOp,
                        VPUIP::NCEClusterTilingOp clusterCopy, mlir::Value newLeftBranch, mlir::Value newRightBranch,
                        const PatternParamsInfo& params, size_t index) const override {
        auto dstBuffer = createNewConcatBuffer(rewriter, subViewOp, clusterCopy,
                                               takeOpLoc(origConcatOp, StringLiteral("buf_{0}"), index));

        // Concat on other axis, so original offset is used
        auto srcOffset = Shape(parseIntArrayAttr<int64_t>(subViewOp.getStaticOffsets()));
        auto createViewBranch = [&](mlir::Value src, int64_t origDimSize, int64_t dstOffsetVal, int64_t srcOffsetVal,
                                    StringRef locSuffix) -> mlir::Value {
            auto copyShape = getShape(subViewOp->getResult(0)).toValues();
            copyShape[params.newConcatDim] = origDimSize;

            Shape newSrcOffset(srcOffset);
            newSrcOffset[params.newConcatDim] = srcOffsetVal;

            Shape dstOffset(SmallVector<int64_t>(copyShape.size(), 0));
            dstOffset[params.newConcatDim] = dstOffsetVal;

            return createNewCopyBranch(rewriter, src, dstBuffer, copyShape, newSrcOffset, dstOffset,
                                       origConcatOp->getLoc(), locSuffix, index);
        };

        auto newLeftViewBranch = createViewBranch(newLeftBranch, params.leftConcatInputSize, /*dstOffsetVal=*/0,
                                                  params.leftViewOffset, "left");
        auto newRightViewBranch = createViewBranch(newRightBranch, params.rightInputSize, params.leftConcatInputSize,
                                                   /*srcOffsetVal=*/0, "right");

        SmallVector<mlir::Value> concatInputs{newLeftViewBranch, newRightViewBranch};
        auto newConcatOp = rewriter.create<VPUIP::ConcatViewOp>(
                takeOpLoc(origConcatOp, StringLiteral("concat_{0}"), index), concatInputs, dstBuffer);
        rewriter.replaceAllUsesWith(clusterCopy->getResult(0), newConcatOp);
        rewriter.eraseOp(clusterCopy);
        rewriter.eraseOp(subViewOp);
    }
};

class SplitUnbalancedDDRConcatOnSameAxis : public SplitUnbalancedDDRConcatBase {
public:
    using SplitUnbalancedDDRConcatBase::SplitUnbalancedDDRConcatBase;

private:
    StringRef getRewriterSuffix() const override {
        return "OnSameAxis";
    }

    bool isSplitSupported(Dim newConcatDim, int64_t tilingDim) const override {
        return newConcatDim.ind() == tilingDim;
    }

    std::pair<mlir::Value, mlir::Operation*> getRightBranchInput(VPUIP::ConcatViewOp concatOp) const override {
        for (auto inputCopy : concatOp.getInputs()) {
            if (auto clusterOp = inputCopy.getDefiningOp<VPUIP::NCEClusterTilingOp>()) {
                bool isValidOp = !mlir::isa<mlir::BlockArgument>(clusterOp->getOperand(0)) &&
                                 clusterOp.getInnerTaskOpOfType<VPUIP::CopyOp>();
                if (isValidOp) {
                    return {clusterOp->getOperand(0), clusterOp};
                }
            }
        }
        return {nullptr, nullptr};
    }

    mlir::Value prepareRightBranch(mlir::PatternRewriter&, mlir::Value rightBranchInput, mlir::Operation*,
                                   VPUIP::PermuteCastOp, mlir::Location) const override {
        return rightBranchInput;
    }

    mlir::Value createNewConcatBuffer(mlir::PatternRewriter& rewriter, VPUIP::SubViewOp,
                                      VPUIP::NCEClusterTilingOp clusterCopy, mlir::Location bufferLoc) const override {
        auto dstBufferType = clusterCopy.getOutputBuffs()[0].getType();
        return rewriter.create<VPURT::AllocDistributed>(bufferLoc, dstBufferType, nullptr, nullptr);
    }

    void rewriteSubview(mlir::PatternRewriter& rewriter, VPUIP::ConcatViewOp origConcatOp, VPUIP::SubViewOp subViewOp,
                        VPUIP::NCEClusterTilingOp clusterCopy, mlir::Value newLeftBranch, mlir::Value newRightBranch,
                        const PatternParamsInfo& params, size_t index) const override {
        auto dstBuffer = createNewConcatBuffer(rewriter, subViewOp, clusterCopy,
                                               takeOpLoc(origConcatOp, StringLiteral("buf_{0}"), index));

        auto newConcatDim = params.newConcatDim;
        // Concat on same axis, so must do manual strided access
        auto srcOffset = Shape(parseIntArrayAttr<int64_t>(subViewOp.getStaticOffsets()));
        auto origConcatDimSize = getShape(origConcatOp->getResult(0))[params.origConcatDim];
        auto viewMultiplier = srcOffset[newConcatDim] / origConcatDimSize;
        mlir::Value newLeftViewBranch = nullptr;
        {
            auto copyShape = getShape(subViewOp->getResult(0)).toValues();
            copyShape[newConcatDim] = params.leftConcatInputSize;

            Shape srcOffset(SmallVector<int64_t>(copyShape.size(), 0));
            srcOffset[newConcatDim] = params.leftInputSize * viewMultiplier + params.leftViewOffset;

            Shape dstOffset(SmallVector<int64_t>(copyShape.size(), 0));
            newLeftViewBranch = createNewCopyBranch(rewriter, newLeftBranch, dstBuffer, copyShape, srcOffset, dstOffset,
                                                    origConcatOp->getLoc(), "left", index);
        }

        mlir::Value newRightViewBranch = nullptr;
        {
            auto baseLoc = origConcatOp->getLoc();
            auto previewShape = getShape(newRightBranch).toValues();
            previewShape[Dim(1)] = params.rightInputSize;
            auto singleAxisView = rewriter.createOrFold<VPUIP::ExtractFlatSliceOp>(
                    appendLoc(baseLoc, "pseudo_dst_view_{1}", index), newRightBranch, viewMultiplier);
            auto normalizedShape = propagateReshapeCast(rewriter, singleAxisView, params.castOp, baseLoc,
                                                        printToString("right_{0}", index));

            auto dstView = rewriter.createOrFold<VPUIP::ExtractFlatSliceOp>(
                    appendLoc(baseLoc, "right_dst_view_{0}", index), dstBuffer, params.leftConcatInputSize);

            auto copyShape = getShape(subViewOp->getResult(0)).toValues();
            copyShape[params.newConcatDim] = params.rightInputSize;

            newRightViewBranch = rewriter.create<VPUIP::CopyOp>(appendLoc(baseLoc, "right_copy_{0}", index),
                                                                normalizedShape, dstView);
        };

        SmallVector<mlir::Value> concatInputs{newLeftViewBranch, newRightViewBranch};
        auto newConcatOp = rewriter.create<VPUIP::ConcatViewOp>(
                takeOpLoc(origConcatOp, StringLiteral("concat_{0}"), index), concatInputs, dstBuffer);
        rewriter.replaceAllUsesWith(clusterCopy->getResult(0), newConcatOp);
        rewriter.eraseOp(clusterCopy);
        rewriter.eraseOp(subViewOp);
    }
};

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
    patterns.add<OptimizeConcatSubviewPattern>(&ctx, _log);
    patterns.add<SplitUnbalancedDDRConcatOnOtherAxis>(&ctx, _log);
    patterns.add<SplitUnbalancedDDRConcatOnSameAxis>(&ctx, _log);

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
