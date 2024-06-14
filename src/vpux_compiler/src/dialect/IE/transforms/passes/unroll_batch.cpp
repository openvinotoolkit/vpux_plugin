//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/unroll_batch.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>

using namespace vpux;

namespace {

SmallVector<mlir::Value> sliceInputs(mlir::PatternRewriter& rewriter, mlir::Operation* origOp, int64_t sliceIdx,
                                     size_t numInputs) {
    const auto operands = origOp->getOperands();
    SmallVector<mlir::Value> slices;
    for (const auto inputIdx : irange(numInputs)) {
        const auto input = operands[inputIdx];
        const auto prevOperands = operands.take_front(inputIdx);
        const auto similarInput = llvm::find(prevOperands, input);
        if (similarInput == prevOperands.end()) {
            const auto shape = getShape(input);
            Shape offsets = Shape(shape.size(), 0);
            offsets[Dim(0)] = checked_cast<int64_t>(sliceIdx);
            const auto offsetsAttr = getIntArrayAttr(rewriter.getContext(), offsets);

            Shape sizes = shape.raw();
            sizes[Dim(0)] = 1;
            const auto sizesAttr = getIntArrayAttr(rewriter.getContext(), sizes);

            const auto subViewOp = rewriter.createOrFold<IE::SliceOp>(origOp->getLoc(), input, offsetsAttr, sizesAttr);
            slices.push_back(subViewOp);
        } else {
            const auto similarSliceIdx = std::distance(prevOperands.begin(), similarInput);
            slices.push_back(slices[similarSliceIdx]);
        }
    }
    return slices;
}

mlir::Value appendOperationsToSlices(mlir::PatternRewriter& rewriter, mlir::Operation* origOp,
                                     mlir::ValueRange slices) {
    const auto origOperands = origOp->getOperands();

    mlir::IRMapping mapper;
    mapper.map(origOperands.take_front(slices.size()), slices);

    auto* newOp = rewriter.clone(*origOp, mapper);
    inferReturnTypes(newOp, InferShapedTypeMode::SHAPE);

    return newOp->getResult(0);
}

}  // namespace

mlir::LogicalResult vpux::IE::genericBatchUnroll(mlir::Operation* origOp, size_t numInputs,
                                                 mlir::PatternRewriter& rewriter) {
    const auto operands = origOp->getOperands();
    VPUX_THROW_WHEN(operands.empty(), "No operands to slice");
    VPUX_THROW_WHEN(origOp->getNumResults() != 1, "Operations with multiple results are not supported");
    VPUX_THROW_UNLESS(operands.size() >= numInputs,
                      "Not enough operands to slice. Not less than {0} expected, but {1} provided", numInputs,
                      operands.size());

    const auto input1 = operands[0];
    const auto input1Shape = getShape(input1);
    const auto rowCount = input1Shape[Dim(0)];
    const auto operandsToSlice = operands.take_front(numInputs);

    const bool isBatchEqual =
            std::all_of(operandsToSlice.begin(), operandsToSlice.end(), [rowCount](mlir::Value value) {
                return getShape(value)[Dim(0)] == rowCount;
            });
    VPUX_THROW_UNLESS(isBatchEqual, "The pass can only slice the inputs with equal batch dimension");

    SmallVector<mlir::Value> slicesToConcat;
    for (const auto sliceIdx : irange(rowCount)) {
        const auto slices = sliceInputs(rewriter, origOp, sliceIdx, numInputs);
        VPUX_THROW_UNLESS(slices.size() == numInputs, "Slices range must contain {0} values, but {1} provided",
                          numInputs, slices.size());

        const auto output = appendOperationsToSlices(rewriter, origOp, slices);
        slicesToConcat.push_back(output);
    }

    rewriter.replaceOpWithNewOp<IE::ConcatOp>(origOp, slicesToConcat, Dim(0).ind());
    return mlir::success();
}

bool vpux::IE::isBatchEqualToOne(const mlir::Value val) {
    const auto inputShape = getShape(val);
    const auto rowCount = inputShape[Dim(0)];
    return rowCount == 1;
}

bool vpux::IE::isShapeRankEqualToZero(const mlir::Value val) {
    const auto inputShape = getShape(val);
    return inputShape.size() == 0;
}

bool vpux::IE::areShapeRanksEqual(const mlir::Value lhs, const mlir::Value rhs) {
    const auto inputShape1 = getShape(lhs);
    const auto inputShape2 = getShape(rhs);
    return inputShape1.size() == inputShape2.size();
}
