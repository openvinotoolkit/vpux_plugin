//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <cstdint>
#include <utility>

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

using namespace vpux;

namespace {

//
// ExtractWeightsAndBiasesFromLSTMSequenceRewriter
//

// The matrix multiplication of inputData and weights, and the addition of biases, can be computed once for the entire
// sequence, as they are not calculated recursively. This rewriter extracts these operations from LSTMSequence to allow
// them to run on the DPU.

class ExtractWeightsAndBiasesFromLSTMSequenceRewriter final : public mlir::OpRewritePattern<IE::LSTMSequenceOp> {
public:
    ExtractWeightsAndBiasesFromLSTMSequenceRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::LSTMSequenceOp>(ctx, benefit), _log(std::move(log)) {
        this->setDebugName("ExtractWeightsAndBiasesFromLSTMSequenceRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::LSTMSequenceOp op, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// The matrix multiplication of inputData and weights, and the addition of biases, can be computed once for the entire
// sequence, as they are not calculated recursively.
// However, when dealing with dynamic input for LSTMSequence, this approach cannot be applied directly because Add and
// MatMul operations do not yet support dynamic shapes fully. Instead, we can execute these operations using an
// upper-bounded input (via the DynamicExtend operation). Subsequently, we can use Slice to remove any unnecessary data
// from the results.
mlir::Value padWeightsAndBiases(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value inputData,
                                mlir::Value newInputData, mlir::Value newWeights, mlir::Value biases,
                                mlir::ArrayAttr axisOneArrayAttr, mlir::MLIRContext* ctx) {
    auto newBiasesOp =
            rewriter.create<IE::UnsqueezeOp>(appendLoc(loc, "_biasesUnsqueeze"), biases, nullptr, axisOneArrayAttr);
    auto padForMatMul = rewriter.create<IE::DynamicExpandOp>(appendLoc(loc, "_padForMatMul"), newInputData);

    auto matMulInputOp =
            rewriter.create<IE::MatMulOp>(appendLoc(loc, "_matMul"), padForMatMul, newWeights, false, true);
    auto addOp = rewriter.create<IE::AddOp>(appendLoc(loc, "_add"), matMulInputOp, newBiasesOp,
                                            IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY), nullptr,
                                            nullptr, nullptr, nullptr);

    auto addShape = to_small_vector(getShape(addOp.getOutput()));
    auto inputDataShape = to_small_vector(getShape(inputData));
    auto addBounds = addShape;

    const auto addShapeRank = checked_cast<int64_t>(addShape.size());
    const auto inputDataShapeRank = checked_cast<int64_t>(inputDataShape.size());

    const auto seqLengthAddIndex = addShapeRank - 2;
    const auto seqLengthDataIndex = inputDataShapeRank - 2;

    addShape[seqLengthAddIndex] = to_small_vector(inputDataShape)[seqLengthDataIndex];
    addBounds[seqLengthAddIndex] = parseIntArrayAttr<int64_t>(getBounds(inputData))[seqLengthDataIndex];

    const auto shapeAttr = getIntArrayAttr(ctx, addShape);
    const auto boundedShapeAttr = getIntArrayAttr(ctx, addBounds);
    const auto dataTypeShapeOf = mlir::TypeAttr::get(getSInt64Type(ctx));

    // In order to apply StridedSlice (to the operations that were performed with upper bounds), we need to determine
    // the correct shape for the output tensor. To achieve this, we will obtain the actual dynamic shape using ShapeOf
    // and use it as the ends input for StridedSlice.

    // %shape_of_cst = IE.ShapeOf(%cst_0) -> 1x2x512x512
    auto shapeOfCst = rewriter.create<IE::ShapeOfOp>(appendLoc(loc, "_shapeOfCst"), newWeights, dataTypeShapeOf);

    // %slice_cst = IE.Slice %shape_of_cst [0] [2] -> 1x2
    auto sliceCst = rewriter.create<IE::SliceOp>(appendLoc(loc, "_sliceCst"), shapeOfCst,
                                                 /*begins=*/rewriter.getI64ArrayAttr({0}),
                                                 /*sizes=*/rewriter.getI64ArrayAttr({2}));

    // %shape_of_0 = IE.ShapeOf(%0) -> 1x1x?x512
    auto shapeOfInput = rewriter.create<IE::ShapeOfOp>(appendLoc(loc, "_shapeOfInput"), newInputData, dataTypeShapeOf);

    // %slice_0 = IE.Slice %shape_of_0 [2] [1] -> ?
    auto sliceInput = rewriter.create<IE::SliceOp>(appendLoc(loc, "_sliceInput"), shapeOfInput,
                                                   /*begins=*/rewriter.getI64ArrayAttr({2}),
                                                   /*sizes=*/rewriter.getI64ArrayAttr({1}));

    // %shape_of_3 = IE.ShapeOf(%3) -> 1x2x35x512
    auto shapeOfAdd = rewriter.create<IE::ShapeOfOp>(appendLoc(loc, "_shapeOfAdd"), addOp.getOutput(), dataTypeShapeOf);

    // %slice_3 = IE.Slice %shape_of_3 [3] [1] -> 512
    auto sliceAdd = rewriter.create<IE::SliceOp>(appendLoc(loc, "_sliceAdd"), shapeOfAdd,
                                                 /*begins=*/rewriter.getI64ArrayAttr({3}),
                                                 /*sizes=*/rewriter.getI64ArrayAttr({1}));

    // %concat = IE.Concat(%slice_cst, %slice_0, %slice_3) -> 1x2x?x512
    const auto axisAttr = getIntAttr(ctx, 0);
    SmallVector<mlir::Value> concatInputs;
    concatInputs.push_back(sliceCst);
    concatInputs.push_back(sliceInput);
    concatInputs.push_back(sliceAdd);
    auto concat = rewriter.create<IE::ConcatOp>(appendLoc(loc, "_concat"), concatInputs, axisAttr);

    // StridedSlice operation
    const SmallVector<int64_t> begins(addShapeRank, 0);
    const SmallVector<int64_t> strides(addShapeRank, 1);

    const auto beginsAttr = getIntArrayAttr(ctx, begins);
    const auto stridesAttr = getIntArrayAttr(ctx, strides);

    const SmallVector<int64_t> empty(addShapeRank, 0);
    const auto emptyAttr = getIntArrayAttr(ctx, empty);

    auto sliceOp = rewriter.create<IE::StridedSliceOp>(appendLoc(loc, "_inputDataSlice"),
                                                       /*data=*/addOp.getOutput(),
                                                       /*begins=*/nullptr,
                                                       /*ends=*/concat.getOutput(),
                                                       /*strides=*/nullptr,
                                                       /*beginsAttr=*/beginsAttr,
                                                       /*endsAttr=*/nullptr,
                                                       /*stridesAttr=*/stridesAttr,
                                                       /*beginMask=*/emptyAttr,
                                                       /*endMask=*/emptyAttr,
                                                       /*newAxisMask=*/emptyAttr,
                                                       /*shrinkAxisMask=*/emptyAttr,
                                                       /*ellipsisMask=*/emptyAttr);

    // StridedSlice infers its output as tensor<?x?x?x?xf16> when any of the begins, ends, or strides are unknown at
    // compile time. DynamicReshape resolves the mismatch between the output of StridedSlice and the input of
    // mlir.return.
    auto reshapedAddOp = rewriter.create<IE::DynamicReshapeOp>(appendLoc(loc, "_reshapedSlice"), sliceOp,
                                                               concat.getOutput(), shapeAttr, boundedShapeAttr);

    return reshapedAddOp;
}

mlir::LogicalResult ExtractWeightsAndBiasesFromLSTMSequenceRewriter::matchAndRewrite(
        IE::LSTMSequenceOp op, mlir::PatternRewriter& rewriter) const {
    auto inputData = op.getInputData();

    const auto weights = op.getWeights();
    const auto biases = op.getBiases();
    if (!weights || !biases) {
        return mlir::failure();
    }

    const auto initialHiddenStateShape = getShape(op.getInitialHiddenState());
    const auto batchSize = initialHiddenStateShape[Dim(0)];
    const auto numDirections = initialHiddenStateShape[Dim(1)];

    const auto ctx = rewriter.getContext();
    const auto loc = op.getLoc();
    const auto axisOne = 1;
    const auto axisZeroArrayAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{0});
    const auto axisOneArrayAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{axisOne});
    const auto isInputDataDynamic = getShape(inputData).isDynamic();

    mlir::Value newInputData;
    if (isInputDataDynamic) {
        auto newInputDataShape = to_small_vector(getShape(inputData));
        auto newInputDataBounds = parseIntArrayAttr<int64_t>(getBounds(inputData));

        newInputDataShape.insert(newInputDataShape.begin() + axisOne, 1);
        newInputDataBounds.insert(newInputDataBounds.begin() + axisOne, 1);

        const auto newInputDataShapeAttr = getIntArrayAttr(ctx, newInputDataShape);
        const auto newInputDataBoundsAttr = getIntArrayAttr(ctx, newInputDataBounds);
        const auto newInputDataShapeRank = checked_cast<int64_t>(newInputDataShape.size());

        const auto dataType = mlir::RankedTensorType::get({newInputDataShapeRank}, getSInt32Type(ctx));
        auto newInputDataShapeValues = IE::replaceDynamicDimsWithValue<int32_t>(newInputDataShape, -1);

        const auto shapeTensor = Const::createConst(rewriter, loc, dataType, ArrayRef(newInputDataShapeValues));
        newInputData =
                rewriter.create<IE::DynamicReshapeOp>(appendLoc(loc, "_inputDataUnsqueeze"), inputData, shapeTensor,
                                                      newInputDataShapeAttr, newInputDataBoundsAttr);
    } else {
        newInputData = rewriter.create<IE::UnsqueezeOp>(appendLoc(loc, "_inputDataUnsqueeze"), inputData, nullptr,
                                                        axisOneArrayAttr);
    }

    mlir::Value newWeights =
            rewriter.create<IE::UnsqueezeOp>(appendLoc(loc, "_weightsUnsqueeze"), weights, nullptr, axisZeroArrayAttr);

    if (numDirections > 1) {
        auto newInputDataShape = Shape(getShape(newInputData));
        newInputDataShape[Dim(1)] = numDirections;
        auto newWeightsShape = Shape(getShape(newWeights));
        newWeightsShape[Dim(0)] = batchSize;

        if (isInputDataDynamic) {
            const auto shapeValues = IE::replaceDynamicDimsWithValue<int32_t>(to_small_vector(newInputDataShape), -1);
            const auto newInputDataShapeRank = checked_cast<int64_t>(newInputDataShape.size());

            const auto dataType = mlir::RankedTensorType::get({newInputDataShapeRank}, getSInt32Type(ctx));
            const auto shapeTensor = Const::createConst(rewriter, loc, dataType, ArrayRef(shapeValues));

            const auto shapeAttr = getIntArrayAttr(ctx, to_small_vector(getShape(newInputData)));
            const auto boundedShapeAttr = getIntArrayAttr(ctx, parseIntArrayAttr<int64_t>(getBounds(newInputData)));

            auto reshapedInput = rewriter.create<IE::DynamicReshapeOp>(appendLoc(loc, "_reshapedInput"), newInputData,
                                                                       shapeTensor, shapeAttr, boundedShapeAttr);

            const auto dataTypeShapeOf = mlir::TypeAttr::get(getSInt64Type(ctx));
            auto shapeOf = rewriter.create<IE::ShapeOfOp>(appendLoc(loc, "_reshapedInputShapeOf"), reshapedInput,
                                                          dataTypeShapeOf);

            newInputData = rewriter.create<IE::DynamicBroadcastOp>(
                    appendLoc(loc, "_inputDataBroadcast"), newInputData, shapeOf.getOutput(), nullptr,
                    IE::BroadcastTypeAttr::get(ctx, IE::BroadcastType::NUMPY), shapeAttr, boundedShapeAttr);
        } else {
            newInputData = rewriter.create<IE::BroadcastOp>(
                    appendLoc(loc, "_inputDataBroadcast"), newInputData,
                    IE::createShapeConstForBroadCast(rewriter, ctx, loc, newInputDataShape), nullptr,
                    IE::BroadcastTypeAttr::get(ctx, IE::BroadcastType::NUMPY));
        }
        newWeights =
                rewriter.create<IE::BroadcastOp>(appendLoc(loc, "_weightsBroadcast"), newWeights,
                                                 IE::createShapeConstForBroadCast(rewriter, ctx, loc, newWeightsShape),
                                                 nullptr, IE::BroadcastTypeAttr::get(ctx, IE::BroadcastType::NUMPY));
    }

    if (isInputDataDynamic) {
        auto reshapedAddOp =
                padWeightsAndBiases(rewriter, loc, inputData, newInputData, newWeights, biases, axisOneArrayAttr, ctx);

        auto newLSTMSequenceOp = rewriter.create<IE::LSTMSequenceOp>(
                loc, reshapedAddOp, op.getInitialHiddenState(), op.getInitialCellState(), nullptr,
                op.getReccurenceWeights(), nullptr, op.getSequenceLengthAttr(), op.getDirectionAttr());

        rewriter.replaceOp(op, newLSTMSequenceOp);
    } else {
        auto newBiasesOp =
                rewriter.create<IE::UnsqueezeOp>(appendLoc(loc, "_biasesUnsqueeze"), biases, nullptr, axisOneArrayAttr);
        auto matMulInputOp =
                rewriter.create<IE::MatMulOp>(appendLoc(loc, "_matMul"), newInputData, newWeights, false, true);
        auto addOp =
                rewriter.create<IE::AddOp>(appendLoc(loc, "_add"), matMulInputOp, newBiasesOp,
                                           IE::AutoBroadcastTypeAttr::get(getContext(), IE::AutoBroadcastType::NUMPY),
                                           nullptr, nullptr, nullptr, nullptr);

        auto newLSTMSequenceOp = rewriter.create<IE::LSTMSequenceOp>(
                loc, addOp, op.getInitialHiddenState(), op.getInitialCellState(), nullptr, op.getReccurenceWeights(),
                nullptr, op.getSequenceLengthAttr(), op.getDirectionAttr());

        rewriter.replaceOp(op, newLSTMSequenceOp);
    }
    return mlir::success();
}

//
// DecomposeLSTMSequenceBidirectionalRewriter
//

// Decompose a bidirectional LSTMSequence into one forward and one reverse operator. It is a preparation step for
// unrolling an LSTMSequence operator to LSTMCell operators and is executed if the operation configuration is not
// supported by the VPU::LSTMSequenceOp.

class DecomposeLSTMSequenceBidirectionalRewriter final : public mlir::OpRewritePattern<IE::LSTMSequenceOp> {
public:
    DecomposeLSTMSequenceBidirectionalRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::LSTMSequenceOp>(ctx, benefit), _log(std::move(log)) {
        this->setDebugName("DecomposeLSTMSequenceBidirectionalRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::LSTMSequenceOp op, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DecomposeLSTMSequenceBidirectionalRewriter::matchAndRewrite(IE::LSTMSequenceOp op,
                                                                                mlir::PatternRewriter& rewriter) const {
    if (VPU::LSTMSequenceOp::isSupported(op)) {
        return mlir::failure();
    }

    const auto direction = op.getDirection();
    if (direction != IE::RNNSequenceDirection::BIDIRECTIONAL) {
        return mlir::failure();
    }

    const auto loc = op.getLoc();
    const auto ctx = rewriter.getContext();

    int splitIdx = 0;
    const auto splitOnDim = [&](mlir::Value input, int64_t dim) -> std::pair<mlir::Value, mlir::Value> {
        if (!input) {
            return {nullptr, nullptr};
        }
        const auto inputShape = getShape(input).raw();

        VPUX_THROW_UNLESS(dim < static_cast<int64_t>(inputShape.size()), "Dim {0} is out of expected range [0, {1}]",
                          dim, inputShape.size() - 1);
        VPUX_THROW_UNLESS(inputShape[dim] == 2, "Expected inputShape[{0}] to be 2, got {1}", dim, inputShape[dim]);

        SmallVector<int64_t> sliceSizes(inputShape);
        sliceSizes[dim] = 1;
        const auto sliceSizesArrayAttr = getIntArrayAttr(ctx, sliceSizes);
        SmallVector<int64_t> sliceOffsets(inputShape.size(), 0);

        const mlir::Value sliceForward =
                rewriter.create<IE::SliceOp>(appendLoc(loc, "_sliceForward_{0}", splitIdx), input,
                                             getIntArrayAttr(ctx, sliceOffsets), sliceSizesArrayAttr);
        sliceOffsets[dim] = 1;
        const mlir::Value sliceReverse =
                rewriter.create<IE::SliceOp>(appendLoc(loc, "_sliceReverse_{0}", splitIdx), input,
                                             getIntArrayAttr(ctx, sliceOffsets), sliceSizesArrayAttr);
        splitIdx++;
        return {sliceForward, sliceReverse};
    };

    const auto [inputDataForward, inputDataReverse] = splitOnDim(op.getInputData(), 1);
    const auto [initialHiddenStateForward, initialHiddenStateReverse] = splitOnDim(op.getInitialHiddenState(), 1);
    const auto [initialCellStateForward, initialCellStateReverse] = splitOnDim(op.getInitialCellState(), 1);
    const auto [weightsForward, weightsReverse] = splitOnDim(op.getWeights(), 0);
    const auto [recurrenceWeightsForward, recurrenceWeightsReverse] = splitOnDim(op.getReccurenceWeights(), 0);
    const auto [biasesForward, biasesReverse] = splitOnDim(op.getBiases(), 0);

    auto lstmSequenceForwardOp = rewriter.create<IE::LSTMSequenceOp>(
            appendLoc(loc, "_forward"), inputDataForward, initialHiddenStateForward, initialCellStateForward,
            weightsForward, recurrenceWeightsForward, biasesForward, op.getSequenceLengthAttr(),
            IE::RNNSequenceDirectionAttr::get(ctx, IE::RNNSequenceDirection::FORWARD));

    auto lstmSequenceReverseOp = rewriter.create<IE::LSTMSequenceOp>(
            appendLoc(loc, "_reverse"), inputDataReverse, initialHiddenStateReverse, initialCellStateReverse,
            weightsReverse, recurrenceWeightsReverse, biasesReverse, op.getSequenceLengthAttr(),
            IE::RNNSequenceDirectionAttr::get(ctx, IE::RNNSequenceDirection::REVERSE));

    auto outputHiddenValuesConcatOp =
            rewriter.create<IE::ConcatOp>(appendLoc(loc, "_hiddenValuesConcat"),
                                          SmallVector<mlir::Value>{lstmSequenceForwardOp.getOutputHiddenValues(),
                                                                   lstmSequenceReverseOp.getOutputHiddenValues()},
                                          Dim(1));
    auto outputHiddenStateConcatOp =
            rewriter.create<IE::ConcatOp>(appendLoc(loc, "_hiddenStateConcat"),
                                          SmallVector<mlir::Value>{lstmSequenceForwardOp.getOutputHiddenState(),
                                                                   lstmSequenceReverseOp.getOutputHiddenState()},
                                          Dim(1));
    auto outputCellStateConcatOp =
            rewriter.create<IE::ConcatOp>(appendLoc(loc, "_cellStateConcat"),
                                          SmallVector<mlir::Value>{lstmSequenceForwardOp.getOutputCellState(),
                                                                   lstmSequenceReverseOp.getOutputCellState()},
                                          Dim(1));

    const SmallVector<mlir::Value> newResults{outputHiddenValuesConcatOp, outputHiddenStateConcatOp,
                                              outputCellStateConcatOp};
    rewriter.replaceOp(op, newResults);

    return mlir::success();
}

//
// UnrollLSTMSequenceToLSTMCellsRewriter
//

// Convert an LSTMSequence operator to LSTMCell operators if the operation configuration is unsupported by the
// VPU::LSTMSequenceOp.

class UnrollLSTMSequenceToLSTMCellsRewriter final : public mlir::OpRewritePattern<IE::LSTMSequenceOp> {
public:
    UnrollLSTMSequenceToLSTMCellsRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::LSTMSequenceOp>(ctx, benefit), _log(std::move(log)) {
        this->setDebugName("UnrollLSTMSequenceToLSTMCellsRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::LSTMSequenceOp op, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UnrollLSTMSequenceToLSTMCellsRewriter::matchAndRewrite(IE::LSTMSequenceOp op,
                                                                           mlir::PatternRewriter& rewriter) const {
    if (VPU::LSTMSequenceOp::isSupported(op)) {
        return mlir::failure();
    }

    const auto direction = op.getDirection();
    VPUX_THROW_WHEN(direction != IE::RNNSequenceDirection::FORWARD && direction != IE::RNNSequenceDirection::REVERSE,
                    "Expected direction to be FORWARD or REVERSE, got {0}", direction);
    const auto isReverseDirection = direction == IE::RNNSequenceDirection::REVERSE;

    const auto ctx = rewriter.getContext();
    const auto loc = op.getLoc();

    const auto axisZeroArrayAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{0});
    const auto axisOneArrayAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{1});

    int squeezeIdx = 0;
    const auto squeezeOnDim = [&](mlir::Value input, const mlir::ArrayAttr& axis) -> mlir::Value {
        if (!input) {
            return nullptr;
        }
        return rewriter.create<IE::SqueezeOp>(appendLoc(loc, "_sqeenze_{0}", squeezeIdx++), input, nullptr, axis);
    };

    const mlir::Value inputData = squeezeOnDim(op.getInputData(), axisOneArrayAttr);
    mlir::Value hiddenState = squeezeOnDim(op.getInitialHiddenState(), axisOneArrayAttr);
    mlir::Value cellState = squeezeOnDim(op.getInitialCellState(), axisOneArrayAttr);
    const mlir::Value weights = squeezeOnDim(op.getWeights(), axisZeroArrayAttr);
    const mlir::Value reccurenceWeights = squeezeOnDim(op.getReccurenceWeights(), axisZeroArrayAttr);
    const mlir::Value biases = squeezeOnDim(op.getBiases(), axisZeroArrayAttr);

    const auto inputDataShape = getShape(inputData).raw();
    VPUX_THROW_UNLESS(inputDataShape.size() == 3, "inputData expected to be of rank 3, got {0}", inputDataShape.size());
    const auto sequenceLenght = op.getSequenceLength().has_value() ? op.getSequenceLength().value() : 1;
    const auto hiddenSizeAttr = getIntAttr(ctx, getShape(hiddenState).back());

    SmallVector<int64_t> sliceOffsets(inputDataShape.size(), 0);
    SmallVector<int64_t> sliceSizes(inputDataShape);
    sliceSizes[1] = 1;
    const auto sliceSizesAttr = getIntArrayAttr(ctx, sliceSizes);

    SmallVector<mlir::Value> lstmCellResults;

    for (int i = 0; i < sequenceLenght; i++) {
        sliceOffsets[1] = isReverseDirection ? sequenceLenght - 1 - i : i;
        auto sliceOp = rewriter.create<IE::SliceOp>(appendLoc(loc, "_slice_{0}", i), inputData,
                                                    getIntArrayAttr(ctx, sliceOffsets), sliceSizesAttr);
        auto sqeezeOp =
                rewriter.create<IE::SqueezeOp>(appendLoc(loc, "_squeeze_{0}", i), sliceOp, nullptr, axisOneArrayAttr);
        auto lstmCellOp =
                rewriter.create<IE::LSTMCellOp>(appendLoc(loc, "_lstmCell_{0}", i), sqeezeOp, hiddenState, cellState,
                                                weights, reccurenceWeights, biases, hiddenSizeAttr);
        auto unsqueezeOp = rewriter.create<IE::UnsqueezeOp>(
                appendLoc(loc, "_unsqueeze_{0}", i), lstmCellOp.getOutputHiddenState(), nullptr, axisOneArrayAttr);

        lstmCellResults.push_back(unsqueezeOp.getOutput());
        hiddenState = lstmCellOp.getOutputHiddenState();
        cellState = lstmCellOp.getOutputCellState();
    }

    if (isReverseDirection) {
        std::reverse(lstmCellResults.begin(), lstmCellResults.end());
    }

    mlir::Value newOutputHiddenValues =
            rewriter.create<IE::ConcatOp>(takeOpLoc(op, "_concat"), lstmCellResults, Dim(1));
    newOutputHiddenValues = rewriter.create<IE::UnsqueezeOp>(takeOpLoc(op, "_unsqueeze"), newOutputHiddenValues,
                                                             nullptr, axisOneArrayAttr);
    const mlir::Value newHiddenState =
            rewriter.create<IE::UnsqueezeOp>(takeOpLoc(op, "_unsqueeze"), hiddenState, nullptr, axisOneArrayAttr);
    const mlir::Value newCellState =
            rewriter.create<IE::UnsqueezeOp>(takeOpLoc(op, "_unsqueeze"), cellState, nullptr, axisOneArrayAttr);

    const SmallVector<mlir::Value> newResults{newOutputHiddenValues, newHiddenState, newCellState};
    rewriter.replaceOp(op, newResults);

    return mlir::success();
}

//
// UnrollLSTMSequenceToLSTMCellsPass
//

class DecomposeLSTMSequencePass final : public IE::DecomposeLSTMSequenceBase<DecomposeLSTMSequencePass> {
public:
    explicit DecomposeLSTMSequencePass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void DecomposeLSTMSequencePass::safeRunOnFunc() {
    auto& ctx = getContext();

    // To explicitly control the patterns exec order to assure dependency
    // benefitLevels[0] is highest benefit level and represent the relative pattern is the first one to run
    const uint32_t levelCount = 3;
    const auto benefitLevels = getBenefitLevels(levelCount);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ExtractWeightsAndBiasesFromLSTMSequenceRewriter>(&ctx, benefitLevels[0], _log);
    patterns.add<DecomposeLSTMSequenceBidirectionalRewriter>(&ctx, benefitLevels[1], _log);
    patterns.add<UnrollLSTMSequenceToLSTMCellsRewriter>(&ctx, benefitLevels[2], _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDecomposeLSTMSequencePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDecomposeLSTMSequencePass(Logger log) {
    return std::make_unique<DecomposeLSTMSequencePass>(std::move(log));
}
