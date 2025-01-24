//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/IE/transposed_convolution_utils.hpp"

#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

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

    return mlir::success();
}

mlir::Value createPadding(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input, Dim axis,
                          int64_t nums, IE::FakeQuantizeOp inputFQ) {
    auto ctx = rewriter.getContext();

    auto inputShape = getShape(input);
    auto offsets = SmallVector<int64_t>(inputShape.size(), 0);
    auto sizes = SmallVector<int64_t>(inputShape.begin(), inputShape.end());
    offsets[axis.ind()] = inputShape[axis] - 1;
    sizes[axis.ind()] = 1;

    auto subSlice = rewriter.create<IE::SliceOp>(appendLoc(loc, "subslice"), input, getIntArrayAttr(ctx, offsets),
                                                 getIntArrayAttr(ctx, sizes))
                            .getResult();
    if (inputFQ != nullptr) {
        subSlice = vpux::IE::createFQ(rewriter, subSlice, inputFQ, takeOpLoc(inputFQ, "fq_in")).getOutput();
    }

    SmallVector<mlir::Value> subSlices;
    subSlices.push_back(input);
    subSlices.insert(subSlices.end(), nums, subSlice);
    auto concatOp = rewriter.create<IE::ConcatOp>(appendLoc(loc, "slices_concat"), subSlices, axis).getOutput();
    if (inputFQ != nullptr) {
        concatOp = vpux::IE::createFQ(rewriter, concatOp, inputFQ, takeOpLoc(inputFQ, "fq_out")).getOutput();
    }

    return concatOp;
}

}  // namespace IE
}  // namespace vpux
