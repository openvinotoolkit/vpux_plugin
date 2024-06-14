//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/optimize_slice_expand.hpp"
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/expand_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/range.hpp"

#include <numeric>

using namespace vpux;

IE::FuseMode vpux::IE::getFuseMode(ShapeRef patternInShape, ShapeRef patternOutShape) {
    VPUX_THROW_UNLESS(patternInShape.size() == patternOutShape.size(),
                      "The size of the input '{0}' and output '{1}' tensors does not match", patternInShape.size(),
                      patternOutShape.size());
    const auto inOutShapes = zip(patternInShape, patternOutShape);
    const auto isAllInShapeLargerThanOut = llvm::all_of(inOutShapes, [](const auto& inOutShape) {
        return std::get<0>(inOutShape) >= std::get<1>(inOutShape);
    });
    return isAllInShapeLargerThanOut ? IE::FuseMode::CONVERT_TO_SLICE : IE::FuseMode::CONVERT_TO_EXPAND;
}

// Pattern 1: 'SliceOp -> Implicit(optional) -> ExpandOp' convert to 'SliceOp' that should has following limitations:
// 1. padBegin < = sliceOffset
// 2. sliceOffset + sliceStaticSize + padEnd < = inputLen
// And we can get:
// newSliceOffset = sliceOffset - padBegin
// newSliceStaticSize = padBegin + sliceStaticSize + padEnd
//
// InData: |------------------------------------|
//                         inputLen
//                                                           InData: |------------------------------------|
// Slice:  |         |------------------|                                         inputLen
//         sliceOffset  sliceStaticSize
//                                                   ->      Slice:  |    |----------------------------|
// Expand:      |----|------------------|----|                   newSliceOffset   newSliceStaticSize
//           padBegin + sliceStaticSize + padEnd
//                                                           OutData:     |----------------------------|
// OutData:     |----------------------------|                                      outputLen
//                         outputLen
//
// Pattern 2: 'SliceOp -> Implicit(optional) -> ExpandOp' convert to 'ExpandOp' that should has following limitations:
// 1. padBegin > = sliceOffset
// 2. sliceOffset + sliceStaticSize + padEnd > = inputLen
// And we can get:
// newPadBegin = padBegin - sliceOffset
// newPadEnd = padEnd - (inputLen - sliceOffset - sliceStaticSize)
//
// InData:       |----------------------------|
//                          inputLen
//                                                           InData:       |----------------------------|
// Slice:        |   |--------------------|                                         inputLen
//           sliceOffset sliceStaticSize
//                                                     ->    Expand:  |----|----------------------------|---|
// Expand:  |--------|--------------------|-------|                newPadBegin        inputLen        newPadEnd
//           padBegin   sliceStaticSize     padEnd
//                                                           OutData: |-------------------------------------|
// OutData: |-------------------------------------|                                    outputLen
//                         outputLen
//
mlir::FailureOr<std::tuple<Shape, Shape, IE::FuseMode>> vpux::IE::getSliceExpandFusedParameters(IE::SliceOp sliceOp,
                                                                                                IE::ExpandOp expandOp) {
    const auto patternInShape = getShape(sliceOp.getSource());
    const auto patternOutShape = getShape(expandOp.getResult());
    const auto rank = patternInShape.size();

    const auto fuseMode = getFuseMode(patternInShape, patternOutShape);

    const auto expandPadsBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBegin());
    const auto expandPadsEnd = parseIntArrayAttr<int64_t>(expandOp.getPadsEnd());
    const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
    const auto sliceStaticSizes = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());

    // CONVERT_TO_SLICE:  the 'firstShapeRef' is 'newSliceOffsets'; the 'secondShapeRef' is 'newSliceStaticSizes'
    // CONVERT_TO_EXPAND: the 'firstShapeRef' is 'newPadsBegin'; the 'secondShapeRef' is 'newPadsEnd'
    SmallVector<int64_t> firstShapeRef(rank, 0);
    SmallVector<int64_t> secondShapeRef(rank, 0);
    for (auto idx : irange(rank)) {
        const auto inputLen = patternInShape[Dim(idx)];
        const auto sliceOffset = sliceOffsets[idx];
        const auto sliceStaticSize = sliceStaticSizes[idx];
        const auto padBegin = expandPadsBegin[idx];
        const auto padEnd = expandPadsEnd[idx];

        const auto outDataMaxRange = sliceOffset + sliceStaticSize + padEnd;
        if (fuseMode == IE::FuseMode::CONVERT_TO_SLICE && padBegin <= sliceOffset && outDataMaxRange <= inputLen) {
            firstShapeRef[idx] = sliceOffset - padBegin;
            secondShapeRef[idx] = padBegin + sliceStaticSize + padEnd;
        } else if (fuseMode == IE::FuseMode::CONVERT_TO_EXPAND && padBegin >= sliceOffset &&
                   outDataMaxRange >= inputLen) {
            firstShapeRef[idx] = padBegin - sliceOffset;
            secondShapeRef[idx] = padEnd - (inputLen - sliceOffset - sliceStaticSize);
        } else {
            return mlir::failure();
        }
    }

    return std::tuple<Shape, Shape, IE::FuseMode>(firstShapeRef, secondShapeRef, fuseMode);
}

// Pattern 1: 'ExpandOp -> Implicit(optional) -> SliceOp' convert to 'SliceOp' that should has following limitations:
// 1. padBegin < = sliceOffset
// 2. padBegin + inputLen > = sliceOffset + sliceStaticSize
// And we can get:
// newSliceOffset = sliceOffset - padBegin
// newSliceStaticSize = sliceStaticSize
//
// InData:       |-----------------|
//                    inputLen
//                                                           InData:       |-----------------|
// Expand:  |----|-----------------|------|                                      inputLen
//         padBegin   inputLen      padEnd
//                                                   ->      Slice:        |     |--------|
// Slice:   |          |--------|                                   newSliceOffset  newSliceStaticSize
//        sliceOffset sliceStaticSize
//                                                           OutData:            |--------|
// OutData:            |--------|                                                 outputLen
//                      outputLen
//
// Pattern 2: 'ExpandOp -> Implicit(optional) -> SliceOp' convert to 'Expand' that should has following limitations:
// 1. padBegin > = sliceOffset
// 2. padBegin + inputLen < = sliceOffset + sliceStaticSize
// And we can get:
// newPadBegin = padBegin - sliceOffset
// newPadEnd = sliceOffset + sliceStaticSize - padBegin - inputLen
//
// InData:       |-----------------|
//                    inputLen
//                                                           InData:       |-----------------|
// Expand:  |----|-----------------|------|                                      inputLen
//         padBegin   inputLen      padEnd
//                                                   ->      Expand:     |-|-----------------|--|
// Slice:   |  |----------------------|                              newPadBegin inputLen newPadEnd
//       sliceOffset sliceStaticSize
//                                                           OutData:    |----------------------|
// OutData:    |----------------------|                                           outputLen
//                     outputLen
//
mlir::FailureOr<std::tuple<Shape, Shape, IE::FuseMode>> vpux::IE::getExpandSliceFusedParameters(IE::ExpandOp expandOp,
                                                                                                IE::SliceOp sliceOp) {
    const auto patternInShape = getShape(expandOp.getInput());
    const auto patternOutShape = getShape(sliceOp.getResult());
    const auto rank = patternInShape.size();

    const auto fuseMode = getFuseMode(patternInShape, patternOutShape);

    const auto expandPadsBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBegin());
    const auto expandPadsEnd = parseIntArrayAttr<int64_t>(expandOp.getPadsEnd());
    const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
    const auto sliceStaticSizes = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());

    SmallVector<int64_t> firstShapeRef(rank, 0);
    SmallVector<int64_t> secondShapeRef(rank, 0);
    for (auto idx : irange(rank)) {
        const auto inputLen = patternInShape[Dim(idx)];
        const auto sliceOffset = sliceOffsets[idx];
        const auto sliceStaticSize = sliceStaticSizes[idx];
        const auto padBegin = expandPadsBegin[idx];

        const auto expandDataRange = padBegin + inputLen;
        const auto sliceDataRange = sliceOffset + sliceStaticSize;
        if (fuseMode == IE::FuseMode::CONVERT_TO_SLICE && padBegin <= sliceOffset &&
            expandDataRange >= sliceDataRange) {
            firstShapeRef[idx] = sliceOffset - padBegin;
            secondShapeRef[idx] = sliceStaticSize;
        } else if (fuseMode == IE::FuseMode::CONVERT_TO_EXPAND && padBegin >= sliceOffset &&
                   expandDataRange <= sliceDataRange) {
            firstShapeRef[idx] = padBegin - sliceOffset;
            secondShapeRef[idx] = sliceOffset + sliceStaticSize - padBegin - inputLen;
        } else {
            return mlir::failure();
        }
    }

    return std::tuple<Shape, Shape, IE::FuseMode>(firstShapeRef, secondShapeRef, fuseMode);
}

//
// OptimizeSliceExpand
//

mlir::LogicalResult vpux::IE::OptimizeSliceExpand::matchAndRewrite(IE::ExpandOp expandOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), expandOp->getName(), expandOp->getLoc());
    const auto innerLog = _log.nest();

    auto sliceOp = expandOp.getInput().getDefiningOp<IE::SliceOp>();
    if (sliceOp == nullptr) {
        innerLog.trace("'Expand' at '{0}' input is not 'SliceOp'", expandOp->getLoc());
        return mlir::failure();
    }

    const auto sliceExpandFusedParameters = getSliceExpandFusedParameters(sliceOp, expandOp);
    if (mlir::failed(sliceExpandFusedParameters)) {
        innerLog.trace("Illegal to fuse 'Slice' at '{0}' and 'Expand' at '{1}'", sliceOp->getLoc(), expandOp->getLoc());
        return mlir::failure();
    }

    // It is specific cases for Eltwise NCE Op
    // This Add can be futher reshaped to avoid expand by AdjustInputShapePass
    // TODO(E#95919): Create Sub Pipeline to check dependency between those two passes
    // In1(1x12x64x64) -> Slice(1x3x64x64) -> Expand(1x16x64x64)
    //                                                           -> Add(1x16x64x64) -> Slice(1x3x64x64)
    // In2(1x12x64x64) -> Slice(1x3x64x64) -> Expand(1x16x64x64)
    auto isEltwiseOp = mlir::isa<IE::AddOp, IE::MultiplyOp>(*(expandOp.getOutput().getUsers().begin()));
    auto eltwiseOp = *(expandOp.getOutput().getUsers().begin());
    auto quantizeCastOp = mlir::dyn_cast_or_null<IE::QuantizeCastOp>(*(expandOp.getOutput().getUsers().begin()));
    if (quantizeCastOp != nullptr) {
        isEltwiseOp = mlir::isa<IE::AddOp, IE::MultiplyOp>(*(quantizeCastOp.getOutput().getUsers().begin()));
        eltwiseOp = *(quantizeCastOp.getOutput().getUsers().begin());
    }
    // E#93789: Follow up task to continue keep slice-expand for Eltwise if expand has multi users
    if (expandOp.getOutput().hasOneUse() && isEltwiseOp) {
        auto newExpandedShapeResult = getShapeCastExpandedShape(eltwiseOp, getShape(expandOp.getOutput()).toValues(),
                                                                getShape(expandOp.getInput()).toValues(), _log.nest());
        if (!mlir::failed(newExpandedShapeResult)) {
            innerLog.trace("Expand channel for Eltwise, skip this optimization");
            return mlir::failure();
        }
    }

    const auto sliceExpandFusedParametersVal = sliceExpandFusedParameters.value();
    const auto padsBeginOrOffsetsAttr =
            getIntArrayAttr(expandOp.getContext(), std::get<0>(sliceExpandFusedParametersVal));
    const auto padsEndOrStaticSizesAttr =
            getIntArrayAttr(expandOp.getContext(), std::get<1>(sliceExpandFusedParametersVal));
    const auto fuseMode = std::get<2>(sliceExpandFusedParametersVal);

    if (fuseMode == IE::FuseMode::CONVERT_TO_EXPAND) {
        innerLog.trace("Convert to 'Expand' completed successfully at '{0}'", expandOp->getLoc());
        rewriter.replaceOpWithNewOp<IE::ExpandOp>(expandOp, sliceOp.getSource(), padsBeginOrOffsetsAttr,
                                                  padsEndOrStaticSizesAttr);
        return mlir::success();
    }

    if (fuseMode == IE::FuseMode::CONVERT_TO_SLICE) {
        innerLog.trace("Convert to 'Slice' completed successfully at '{0}'", expandOp->getLoc());
        rewriter.replaceOpWithNewOp<IE::SliceOp>(expandOp, sliceOp.getSource(), padsBeginOrOffsetsAttr,
                                                 padsEndOrStaticSizesAttr);
        return mlir::success();
    }

    return mlir::failure();
}

//
// OptimizeExpandSlice
//

mlir::LogicalResult vpux::IE::OptimizeExpandSlice::matchAndRewrite(IE::ExpandOp expandOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), expandOp->getName(), expandOp->getLoc());
    const auto innerLog = _log.nest();

    auto sliceOp = mlir::dyn_cast<IE::SliceOp>(*expandOp.getOutput().getUsers().begin());

    if (sliceOp == nullptr) {
        innerLog.trace("'Expand' at '{0}' user is not 'SliceOp'", expandOp->getLoc());
        return mlir::failure();
    }

    const auto expandSliceFusedParameters = getExpandSliceFusedParameters(expandOp, sliceOp);
    if (mlir::failed(expandSliceFusedParameters)) {
        innerLog.trace("Illegal to fuse 'Expand' at '{0}' and 'Slice' at '{1}'", expandOp->getLoc(), sliceOp->getLoc());
        return mlir::failure();
    }

    const auto expandSliceFusedParametersVal = expandSliceFusedParameters.value();
    const auto padsBeginOrOffsetsAttr =
            getIntArrayAttr(expandOp.getContext(), std::get<0>(expandSliceFusedParametersVal));
    const auto padsEndOrStaticSizesAttr =
            getIntArrayAttr(expandOp.getContext(), std::get<1>(expandSliceFusedParametersVal));
    const auto fuseMode = std::get<2>(expandSliceFusedParametersVal);

    if (fuseMode == IE::FuseMode::CONVERT_TO_EXPAND) {
        innerLog.trace("Convert to 'Expand' completed successfully at '{0}'", expandOp->getLoc());
        rewriter.replaceOpWithNewOp<IE::ExpandOp>(sliceOp, expandOp.getInput(), padsBeginOrOffsetsAttr,
                                                  padsEndOrStaticSizesAttr);
        return mlir::success();
    }

    if (fuseMode == IE::FuseMode::CONVERT_TO_SLICE) {
        innerLog.trace("Convert to 'Slice' completed successfully at '{0}'", expandOp->getLoc());
        rewriter.replaceOpWithNewOp<IE::SliceOp>(sliceOp, expandOp.getInput(), padsBeginOrOffsetsAttr,
                                                 padsEndOrStaticSizesAttr);
        return mlir::success();
    }

    return mlir::failure();
}

//
// OptimizeSliceImplicitExpand
//

mlir::LogicalResult vpux::IE::genericOptimizeSliceImplicitExpand(IE::ExpandOp expandOp, mlir::Operation* implicitOp,
                                                                 bool hasCalculationCost,
                                                                 mlir::PatternRewriter& rewriter, Logger innerLog) {
    if (implicitOp == nullptr || implicitOp->getNumOperands() != 1 || implicitOp->getNumResults() != 1 ||
        !implicitOp->hasOneUse()) {
        return mlir::failure();
    }

    auto sliceOp = implicitOp->getOperand(0).getDefiningOp<IE::SliceOp>();
    if (sliceOp == nullptr) {
        innerLog.trace("Cannot get 'Slice' before '{0}'", implicitOp->getName());
        return mlir::failure();
    }

    const auto patternInShape = getShape(sliceOp.getSource());
    const auto patternOutShape = getShape(expandOp.getResult());
    // If the implicitOp has calculation cost
    // Only consider the 'slice' and 'expand' can be completely eliminated currently
    // Otherwise not ensure for case that reserve one 'slice' or 'expand' will get the performance benefit
    // Due to the computational size of the SW layer become larger
    // It is possible to remove restrictions on SW layers that has the calculation cost in the future
    // depend on the execution efficiency
    if (hasCalculationCost && patternInShape != patternOutShape) {
        innerLog.trace("'{0}' has calculation cost and 'Slice' and 'Expand' cannot be completely eliminated",
                       implicitOp->getName());
        return mlir::failure();
    }

    const auto sliceExpandFusedParameters = getSliceExpandFusedParameters(sliceOp, expandOp);
    if (mlir::failed(sliceExpandFusedParameters)) {
        innerLog.trace("Illegal to fuse Slice at '{0}' and Expand at '{1}'", sliceOp->getLoc(), expandOp->getLoc());
        return mlir::failure();
    }

    const auto sliceExpandFusedParametersVal = sliceExpandFusedParameters.value();
    const auto padsBeginOrOffsetsAttr =
            getIntArrayAttr(expandOp.getContext(), std::get<0>(sliceExpandFusedParametersVal));
    const auto padsEndOrStaticSizesAttr =
            getIntArrayAttr(expandOp.getContext(), std::get<1>(sliceExpandFusedParametersVal));
    const auto fuseMode = std::get<2>(sliceExpandFusedParametersVal);

    if (fuseMode == IE::FuseMode::CONVERT_TO_EXPAND) {
        innerLog.trace("Convert to 'Expand' completed successfully at '{0}'", expandOp->getLoc());
        rewriter.setInsertionPointAfter(implicitOp);
        implicitOp->getOpOperand(0).set(sliceOp.getSource());
        vpux::inferReturnTypes(implicitOp, vpux::InferShapedTypeMode::SHAPE);
        rewriter.replaceOpWithNewOp<IE::ExpandOp>(expandOp, implicitOp->getResults()[0], padsBeginOrOffsetsAttr,
                                                  padsEndOrStaticSizesAttr);
        return mlir::success();
    }

    if (fuseMode == IE::FuseMode::CONVERT_TO_SLICE) {
        innerLog.trace("Convert to 'Slice' completed successfully at '{0}'", expandOp->getLoc());
        rewriter.setInsertionPoint(implicitOp);
        auto newSliceOp = rewriter.create<IE::SliceOp>(expandOp.getLoc(), sliceOp.getSource(), padsBeginOrOffsetsAttr,
                                                       padsEndOrStaticSizesAttr);
        implicitOp->getOpOperand(0).set(newSliceOp.getResult());
        vpux::inferReturnTypes(implicitOp, vpux::InferShapedTypeMode::SHAPE);
        expandOp->replaceAllUsesWith(implicitOp);
        rewriter.eraseOp(expandOp);
        return mlir::success();
    }

    return mlir::failure();
}

//
// OptimizeSliceConcatExpand
//

namespace {

bool isSliceAxisInLastDim(IE::SliceOp sliceOp, Dim dim) {
    auto inType = sliceOp.getSource().getType().cast<vpux::NDTypeInterface>();
    auto inRank = inType.getRank();
    auto dimsOrder = inType.getDimsOrder();
    return dimsOrder.toMemDim(dim).ind() == inRank - 1;
}

SmallVector<Const::DeclareOp> getAllConstInputOp(IE::ConcatOp origOp) {
    mlir::SmallVector<Const::DeclareOp> inputOps;
    for (auto preOps : origOp.getInputs()) {
        auto constOp = preOps.getDefiningOp<Const::DeclareOp>();

        if (constOp != nullptr) {
            inputOps.emplace_back(constOp);
        }
    }
    return inputOps;
}

}  // namespace

mlir::LogicalResult vpux::IE::OptimizeSliceConcatExpand::matchAndRewrite(IE::ExpandOp expandOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), expandOp->getName(), expandOp->getLoc());
    const auto innerLog = _log.nest();

    auto concatOp = expandOp.getInput().getDefiningOp<IE::ConcatOp>();
    if (concatOp == nullptr || concatOp->getNumResults() != 1 || !concatOp->hasOneUse()) {
        innerLog.trace("'Expand' at '{0}' input is not 'Concat' or 'Concat' has more than one users",
                       expandOp->getLoc());
        return mlir::failure();
    }

    SmallVector<Dim> sliceAxes;
    SmallVector<std::pair<int32_t, IE::SliceOp>> sliceOpInfos;
    for (const auto& concatInput : concatOp.getInputs() | indexed) {
        auto sliceOp = concatInput.value().getDefiningOp<IE::SliceOp>();
        if (sliceOp == nullptr) {
            continue;
        }

        auto sliceAxis = vpux::IE::getSingleDiffAxis(getShape(sliceOp.getSource()), getShape(sliceOp.getResult()));
        if (!sliceAxis.has_value()) {
            return mlir::failure();
        }

        if (sliceAxes.empty() || sliceAxis.value() != sliceAxes.back()) {
            sliceAxes.push_back(sliceAxis.value());
        }

        sliceOpInfos.push_back(std::pair<int32_t, IE::SliceOp>(concatInput.index(), sliceOp));
    }

    const auto concatAxis = getConcatAxis(concatOp);
    const auto expandAxis = getExpandAxis(expandOp);

    if (sliceAxes.size() != 1 || !concatAxis.has_value() || !expandAxis.has_value()) {
        return mlir::failure();
    }

    const auto sliceAxisVal = sliceAxes.front();
    const auto concatAxisVal = concatAxis.value();
    const auto expandAxisVal = expandAxis.value();

    if (sliceAxisVal != expandAxisVal) {
        innerLog.trace("'Slice' axis should same with 'Expand' axis, but got '{0}' and '{1}'", sliceAxisVal,
                       expandAxisVal);
        return mlir::failure();
    }

    const auto expandOutShape = to_small_vector(getShape(expandOp.getResult()));
    const auto expandPadsBegin = parseIntArrayAttr<int64_t>(expandOp.getPadsBegin());
    const auto expandPadsEnd = parseIntArrayAttr<int64_t>(expandOp.getPadsEnd());
    // Only consider the 'slice' and 'expand' can be completely eliminated currently
    // TODO(E#95438): Remove part of 'slice' or 'expand' Op
    SmallVector<mlir::Value> newConcatInputs;
    const auto checkDim = sliceAxisVal.ind();
    if (concatAxisVal != sliceAxisVal) {
        const auto isLegalSliceOp = [&](const auto& sliceOpInfo) {
            auto sliceOp = sliceOpInfo.second;
            const auto sliceInShape = to_small_vector(getShape(sliceOp.getSource()));
            const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
            return sliceOffsets[checkDim] == expandPadsBegin[checkDim] &&
                   sliceInShape[checkDim] == expandOutShape[checkDim];
        };

        if (concatOp.getInputs().size() != sliceOpInfos.size() || !llvm::all_of(sliceOpInfos, isLegalSliceOp)) {
            innerLog.trace("Ilegal pattern at '{0}'", expandOp->getLoc());
            return mlir::failure();
        }
    }

    if (concatAxisVal == sliceAxisVal) {
        const auto isLegalSliceOp = [&](const auto& sliceOpInfo) {
            auto inputIdx = sliceOpInfo.first;
            auto sliceOp = sliceOpInfo.second;
            const auto sliceInShape = to_small_vector(getShape(sliceOp.getSource()));
            const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());
            const auto sliceStaticSizes = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());
            if (inputIdx == 0) {
                return sliceOffsets[checkDim] == expandPadsBegin[checkDim] &&
                       sliceInShape[checkDim] == sliceOffsets[checkDim] + sliceStaticSizes[checkDim];
            } else if (inputIdx == checked_cast<int64_t>(concatOp.getInputs().size()) - 1) {
                return sliceOffsets[checkDim] == 0 &&
                       sliceInShape[checkDim] == expandPadsEnd[checkDim] + sliceStaticSizes[checkDim];
            } else {
                return false;
            }
        };

        if (!llvm::all_of(sliceOpInfos, isLegalSliceOp)) {
            innerLog.trace("Ilegal pattern at '{0}'", expandOp->getLoc());
            return mlir::failure();
        }
    }

    for (const auto& concatInput : concatOp.getInputs()) {
        if (auto sliceOp = concatInput.getDefiningOp<IE::SliceOp>()) {
            newConcatInputs.push_back(sliceOp.getSource());
        } else {
            newConcatInputs.push_back(concatInput);
        }
    }

    innerLog.trace("Optimization completed successfully at '{0}'", expandOp->getLoc());
    rewriter.replaceOpWithNewOp<IE::ConcatOp>(expandOp, newConcatInputs, concatAxisVal);
    return mlir::success();
}

Const::DeclareOp createNewConstInput(mlir::Value concatInput, Dim expandAxisVal, ShapeRef expandInShape,
                                     mlir::PatternRewriter& rewriter) {
    auto concatConstInput = concatInput.getDefiningOp<Const::DeclareOp>();
    const auto constInputshape = getShape(concatConstInput);

    Shape targetShape(constInputshape.size());

    for (auto ind : irange(targetShape.size())) {
        const auto d = Dim(ind);
        targetShape[d] = d == expandAxisVal ? expandInShape[expandAxisVal] : constInputshape[d];
    }

    auto contentAttr = concatConstInput.getContentAttr();
    auto baseContent = contentAttr.getBaseContent();
    auto newConstOutputType = concatConstInput.getOutput().getType().cast<vpux::NDTypeInterface>();
    newConstOutputType = newConstOutputType.changeShape(targetShape);

    Const::ContentAttr newContentAttr = Const::ContentAttr::get(baseContent);
    for (auto& attr : contentAttr.getTransformations()) {
        if (attr.isa<Const::PadWithZeroAttr>()) {
            return nullptr;
        }
        auto broadCastAttr = attr.dyn_cast_or_null<Const::BroadcastAttr>();
        if (broadCastAttr != nullptr) {
            // If the BroadcastAttr Axis is the same to taget Axis, could not handle.
            if (broadCastAttr.getAxis().getValue() == expandAxisVal.ind()) {
                return nullptr;
            }
        }
        newContentAttr = Const::ContentAttr::addTransformation(newContentAttr, attr);
    }

    newContentAttr = newContentAttr.broadcast(expandAxisVal, targetShape[expandAxisVal]);
    return rewriter.create<Const::DeclareOp>(concatConstInput.getLoc(), newConstOutputType, newContentAttr);
}

// For the pattern slice -> concat -> concat -> expand
// Opimize it to concat -> concat with channel not changed.
// This is a typical pattern after handle large padds pass.
mlir::LogicalResult vpux::IE::OptimizeSliceTwoConcatsExpand::matchAndRewrite(IE::ExpandOp expandOp,
                                                                             mlir::PatternRewriter& rewriter) const {
    const auto innerLog = _log.nest();
    innerLog.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), expandOp->getName(), expandOp->getLoc());

    auto concatOp = expandOp.getInput().getDefiningOp<IE::ConcatOp>();
    if (concatOp == nullptr || concatOp->getNumResults() != 1 || !concatOp->hasOneUse()) {
        innerLog.trace("'Expand' at '{0}' input is not 'Concat' or 'Concat' has more than one users",
                       expandOp->getLoc());
        return mlir::failure();
    }

    const auto patternCheck = [](IE::ConcatOp concatOp) {
        // The const input needs to be splat, otherwise could not handle the const expand.
        const auto isSplatConcatConstInputs = [](Const::DeclareOp constOp) {
            return IE::isBaseContentSplat(constOp);
        };
        // Only handle concat with only one non-const input
        auto concatOpConstInputs = getAllConstInputOp(concatOp);
        if (concatOpConstInputs.size() != concatOp.getInputs().size() - 1) {
            return false;
        }
        if (!llvm::all_of(concatOpConstInputs, isSplatConcatConstInputs)) {
            return false;
        }
        return true;
    };

    if (!patternCheck(concatOp)) {
        return mlir::failure();
    }

    // Find the second Concat Op
    SmallVector<IE::ConcatOp> concatSecondOpCandidates;
    for (const auto& concatInput : concatOp.getInputs() | indexed) {
        auto concatOplocal = concatInput.value().getDefiningOp<IE::ConcatOp>();
        if (concatOplocal == nullptr) {
            continue;
        }
        concatSecondOpCandidates.push_back(concatOplocal);
    }
    // Only handle concate -> concate case.
    if (concatSecondOpCandidates.size() != 1) {
        return mlir::failure();
    }

    auto concatSecondOp = concatSecondOpCandidates.front();

    if (!patternCheck(concatSecondOp)) {
        return mlir::failure();
    }

    SmallVector<IE::SliceOp> sliceOpCandidate;
    for (const auto& concatInput : concatSecondOp.getInputs() | indexed) {
        auto sliceOp = concatInput.value().getDefiningOp<IE::SliceOp>();
        if (sliceOp == nullptr) {
            continue;
        }
        sliceOpCandidate.push_back(sliceOp);
    }
    if (sliceOpCandidate.size() != 1) {
        return mlir::failure();
    }

    const auto concatAxis = getConcatAxis(concatOp);
    const auto concatSecondOpAxis = getConcatAxis(concatSecondOp);
    auto sliceOpCandidateFront = sliceOpCandidate.front();
    const auto sliceAxis = vpux::IE::getSingleDiffAxis(getShape(sliceOpCandidateFront.getSource()),
                                                       getShape(sliceOpCandidateFront.getResult()));
    const auto expandAxis = getExpandAxis(expandOp);
    const auto expandAxisVal = expandAxis.value();

    if (!sliceAxis.has_value() || !concatAxis.has_value() || !expandAxis.has_value() ||
        !concatSecondOpAxis.has_value() || !isSliceAxisInLastDim(sliceOpCandidateFront, sliceAxis.value())) {
        return mlir::failure();
    }

    if (sliceAxis.value() != expandAxisVal) {
        innerLog.trace("'Slice' axis should same with 'Expand' axis, but got '{0}' and '{1}'", sliceAxis.value(),
                       expandAxisVal);
        return mlir::failure();
    }
    if (sliceAxis.value() == concatAxis.value()) {
        innerLog.trace("'Slice' axis should be different with 'Concat' axis, but got '{0}' and '{1}'",
                       sliceAxis.value(), concatAxis.value());
        return mlir::failure();
    }
    if (sliceAxis.value() == concatSecondOpAxis.value()) {
        innerLog.trace("'Slice' axis should be different with second 'Concat' axis, but got '{0}' and '{1}'",
                       sliceAxis.value(), concatSecondOpAxis.value());
        return mlir::failure();
    }

    SmallVector<mlir::Value> newConcatInputs;
    const auto expandInShape = getShape(expandOp);
    for (const auto& concatInput : concatSecondOp.getInputs()) {
        if (auto sliceOp = concatInput.getDefiningOp<IE::SliceOp>()) {
            newConcatInputs.push_back(sliceOp.getSource());
        } else {
            auto newConstInput = createNewConstInput(concatInput, expandAxisVal, expandInShape, rewriter);
            if (newConstInput == nullptr) {
                return mlir::failure();
            }
            newConcatInputs.push_back(newConstInput);
        }
    }

    auto newConcatSecondOp =
            rewriter.create<IE::ConcatOp>(concatSecondOp.getLoc(), newConcatInputs, concatSecondOpAxis.value());

    newConcatInputs.clear();
    for (const auto& concatInput : concatOp.getInputs()) {
        if (auto concatOpLocal = concatInput.getDefiningOp<IE::ConcatOp>()) {
            newConcatInputs.push_back(newConcatSecondOp);
        } else {
            auto newConstInput = createNewConstInput(concatInput, expandAxisVal, expandInShape, rewriter);
            if (newConstInput == nullptr) {
                return mlir::failure();
            }
            newConcatInputs.push_back(newConstInput);
        }
    }

    innerLog.trace("Optimization completed successfully at '{0}'", expandOp->getLoc());
    rewriter.replaceOpWithNewOp<IE::ConcatOp>(expandOp, newConcatInputs, concatAxis.value());
    return mlir::success();
}

//
// OptimizeSliceShapeCastExpand
//

// Only consider a simple pattern for now:
//   - input/output shape of ShapeCast has ony one dimension at which the shape is not one,
//     e.g. <Nx1x1x1>, <1x1xHx1> or <1x1x1xW>
//   - ExpandOp also has the same input/output shape pattern as ShapeCast
//
// It guarantees that ExpandOp can expands tensor at the correct axis after swap
// ShapeCast and Expand.

bool canSwapShapeCastAndExpand(IE::ShapeCastOp shapeCastOp, IE::ExpandOp expandOp) {
    const auto shapeNotOne = [](auto dimShape) -> bool {
        return dimShape != 1;
    };

    const auto isOpSingleShape = [&](ShapeRef inputShape, ShapeRef outputShape) -> bool {
        const auto isInputSingleDim = llvm::count_if(inputShape, shapeNotOne) == 1;
        const auto isOutputSingleDim = llvm::count_if(outputShape, shapeNotOne) == 1;
        return isInputSingleDim && isOutputSingleDim;
    };

    return isOpSingleShape(getShape(shapeCastOp.getSource()), getShape(shapeCastOp.getResult())) &&
           isOpSingleShape(getShape(expandOp.getInput()), getShape(expandOp.getResult()));
}

bool canEliminateSliceExpand(IE::ShapeCastOp shapeCastOp, IE::ExpandOp expandOp, ShapeRef sliceInputShape) {
    if (!canSwapShapeCastAndExpand(shapeCastOp, expandOp)) {
        return false;
    }

    // check if the input type of SliceOp and the output type of Expand are the same,
    // if yes, then Slice and Expand can be eliminated.
    // e.g.
    // <1x80x1x1> -> Slice -> <1x72x1x1> -> Sigmoid -> <1x72x1x1> -> ShapeCast -> <72x1x1x1> -> Expand -> <80x1x1x1>
    //
    // The new Expand output type is <1x80x1x1> which is the same as Slice input after swap ShapeCast and Expand
    // <1x80x1x1> -> Slice -> <1x72x1x1> -> Sigmoid -> <1x72x1x1> -> Expand -> <1x80x1x1> -> ShapeCast -> <80x1x1x1>
    //
    const auto shapeNotOne = [](auto dimShape) -> bool {
        return dimShape != 1;
    };

    const auto getDimShapeNotOne = [&](ShapeRef shape) {
        const auto shapeIt = llvm::find_if(shape, shapeNotOne);
        VPUX_THROW_WHEN(shapeIt == shape.end(), "ilegal shape {0}", shape);
        return std::distance(shape.begin(), shapeIt);
    };

    const auto shapeCastInputShape = getShape(shapeCastOp.getSource());
    const auto shapeCastInputShapeDim = getDimShapeNotOne(shapeCastInputShape);
    const auto shapeCastOutputShape = getShape(shapeCastOp.getResult());
    const auto shapeCastOutputShapeDim = getDimShapeNotOne(shapeCastOutputShape);

    const auto expandInputShape = getShape(expandOp.getInput());
    const auto expandInputShapeDim = getDimShapeNotOne(expandInputShape);
    const auto expandOutputShape = getShape(expandOp.getResult());
    const auto expandOutputShapeDim = getDimShapeNotOne(expandOutputShape);

    VPUX_THROW_UNLESS(expandInputShapeDim == expandOutputShapeDim, "not expand at the same axis: {0}, {1}",
                      expandInputShapeDim, expandOutputShapeDim);
    VPUX_THROW_UNLESS(shapeCastOutputShapeDim == expandInputShapeDim, "{0} not expand at the ShapeCast axis {1}",
                      expandInputShapeDim, shapeCastOutputShapeDim);

    auto expandOutShapeAfterSwap = Shape(to_small_vector(shapeCastInputShape));
    expandOutShapeAfterSwap[Dim(shapeCastInputShapeDim)] = expandOutputShape[Dim(expandOutputShapeDim)];

    return expandOutShapeAfterSwap == sliceInputShape;
}

// optimize the pattern below:
//   ->Slice->EltwiseLikeSW->ShapeCast->Expand
// to
//   ->EltwiseLikeSW->ShapeCast

mlir::LogicalResult vpux::IE::genericOptimizeSliceImplicitShapeCastExpand(IE::ExpandOp origOp,
                                                                          IE::ShapeCastOp shapeCastOp,
                                                                          mlir::Operation* implicitOp,
                                                                          mlir::PatternRewriter& rewriter,
                                                                          Logger innerLog) {
    if (implicitOp == nullptr || implicitOp->getNumOperands() != 1 || implicitOp->getNumResults() != 1 ||
        !implicitOp->hasOneUse()) {
        return mlir::failure();
    }

    auto sliceOp = implicitOp->getOperand(0).getDefiningOp<IE::SliceOp>();
    if (sliceOp == nullptr) {
        innerLog.trace("Cannot get 'Slice' before '{0}'", implicitOp->getName());
        return mlir::failure();
    }
    if (!sliceOp->hasOneUse()) {
        return mlir::failure();
    }

    if (!canEliminateSliceExpand(shapeCastOp, origOp, getShape(sliceOp.getSource()))) {
        return mlir::failure();
    }

    // found the beneficial pattern and create new ops:
    //     EltwiseLikeSW->ShapeCast
    //

    implicitOp->getOpOperand(0).set(sliceOp.getSource());
    vpux::inferReturnTypes(implicitOp, vpux::InferShapedTypeMode::SHAPE);

    rewriter.setInsertionPointAfter(implicitOp);
    const auto newShapeCastOutShape = getShape(origOp.getResult());
    auto newShapeCastOp = rewriter.create<IE::ShapeCastOp>(origOp->getLoc(), origOp.getType(), implicitOp->getResult(0),
                                                           getIntArrayAttr(origOp.getContext(), newShapeCastOutShape));

    rewriter.replaceOp(origOp, newShapeCastOp.getResult());
    return mlir::success();
}
