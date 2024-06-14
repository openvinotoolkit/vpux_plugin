//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/IE/transposed_convolution_utils.hpp"

namespace vpux {
namespace IE {
// Checks whether the TransposedConvolution filter is a constant or a FakeQuantize with a constant input
mlir::FailureOr<Const::DeclareOp> getConstFilter(IE::TransposedConvolutionOp transposedConv) {
    if (auto filterFq = transposedConv.getFilter().getDefiningOp<IE::FakeQuantizeOp>()) {
        if (auto filterConst = filterFq.getInput().getDefiningOp<Const::DeclareOp>()) {
            return filterConst;
        }
    } else if (auto filterDeq = transposedConv.getFilter().getDefiningOp<IE::DequantizeOp>()) {
        if (auto filterConst = filterDeq.getInput().getDefiningOp<Const::DeclareOp>()) {
            return filterConst;
        }
    } else if (auto filterConst = transposedConv.getFilter().getDefiningOp<Const::DeclareOp>()) {
        return filterConst;
    }
    return mlir::failure();
}

mlir::LogicalResult canConvertTransposedConvToConv(IE::TransposedConvolutionOp transposedConv) {
    if (getShape(transposedConv.getInput()).size() != 4) {
        return mlir::failure();
    }

    if (mlir::failed(IE::getConstFilter(transposedConv))) {
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult canConvertGroupTransposedConvToGroupConv(IE::GroupTransposedConvolutionOp groupTransposedConv) {
    if (getShape(groupTransposedConv.getInput()).size() != 4) {
        return mlir::failure();
    }

    // Const::DeclareOp - IE::FakeQuantizeOp filter is not handled
    if (!mlir::isa<Const::DeclareOp>(groupTransposedConv.getFilter().getDefiningOp())) {
        return mlir::failure();
    }

    return mlir::success();
}

auto createFQ(mlir::PatternRewriter& rewriter, mlir::Value input, IE::FakeQuantizeOp fq) {
    const auto outputType = fq.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto newOutputType = outputType.changeShape(getShape(input));
    return rewriter
            .create<IE::FakeQuantizeOp>(fq.getLoc(), newOutputType, input, fq.getInputLow(), fq.getInputHigh(),
                                        fq.getOutputLow(), fq.getOutputHigh(), fq.getLevelsAttr(),
                                        fq.getLowFpTypeAttr(), fq.getAutoBroadcast())
            .getOutput();
}

mlir::Value createPadding(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input, Dim axis,
                          int64_t nums, IE::FakeQuantizeOp inputFQ) {
    auto ctx = rewriter.getContext();

    auto inputShape = getShape(input);
    auto offsets = SmallVector<int64_t>(inputShape.size(), 0);
    auto sizes = SmallVector<int64_t>(inputShape.begin(), inputShape.end());
    offsets[axis.ind()] = inputShape[axis] - 1;
    sizes[axis.ind()] = 1;

    auto subSlice = rewriter.create<IE::SliceOp>(loc, input, getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, sizes))
                            .getResult();
    if (inputFQ != nullptr) {
        subSlice = createFQ(rewriter, subSlice, inputFQ);
    }

    SmallVector<mlir::Value> subSlices;
    subSlices.push_back(input);
    subSlices.insert(subSlices.end(), nums, subSlice);
    auto concatOp = rewriter.create<IE::ConcatOp>(loc, subSlices, axis).getOutput();
    if (inputFQ != nullptr) {
        concatOp = createFQ(rewriter, concatOp, inputFQ);
    }

    return concatOp;
}

}  // namespace IE
}  // namespace vpux
