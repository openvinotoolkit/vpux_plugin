//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/type_traits.hpp"

namespace vpux {
namespace IE {

struct FFTParams final {
    SmallVector<int64_t> axes;
    SmallVector<int64_t> signalSize;
};

template <typename T>
mlir::FailureOr<FFTParams> fftExtractParams(mlir::Location loc, T op, bool complexInputType = true) {
    mlir::FailureOr<SmallVector<int64_t>> axes;
    mlir::FailureOr<SmallVector<int64_t>> signalSize;
    if (op.getAxesAttr().has_value()) {
        axes = parseIntArrayAttr<int64_t>(op.getAxesAttr().value());
    } else if (op.getAxes() != nullptr) {
        axes = IE::constInputToData(loc, op.getAxes());
        if (mlir::failed(axes)) {
            return errorAt(loc, "Only constant input is supported for axes");
        }
        auto inType = op.getInput().getType().template dyn_cast<mlir::ShapedType>();
        const auto inRank = inType.getRank();
        auto axesVal = axes.value();
        // DFT, IDFT and IRDFT contain complex data type, represented as tensor with 1 more dimension not consider in
        // parameters
        int64_t decreaseRank = 0;
        if (complexInputType) {
            decreaseRank = 1;
        }
        for (size_t i = 0; i < axesVal.size(); ++i) {
            if (axesVal[i] < 0) {
                axesVal[i] = inRank - decreaseRank + axesVal[i];
            }
        }
        axes = axesVal;
    } else {
        return errorAt(loc, "Axes should be provided as attribute or input constant tensor");
    }

    if (op.getSignalSize() != nullptr) {
        signalSize = IE::constInputToData(loc, op.getSignalSize());
        if (mlir::failed(signalSize)) {
            return errorAt(loc, "Only constant input is supported for signal_size");
        }
    } else {
        if (op.getSignalSizeAttr().has_value()) {
            signalSize = parseIntArrayAttr<int64_t>(op.getSignalSizeAttr().value());
        } else {
            auto axesSize = axes.value().size();
            signalSize = SmallVector<int64_t>(axesSize, -1);
        }
    }
    if (signalSize.value().size() != axes.value().size()) {
        return errorAt(loc, "Axes and signal_size vector should be provided with same size.");
    }
    return FFTParams{std::move(axes.value()), std::move(signalSize.value())};
}

template <class T>
class FftConvertConstToAttrAndNormalize final : public mlir::OpRewritePattern<T> {
public:
    using mlir::OpRewritePattern<T>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(T op, mlir::PatternRewriter& rewriter) const final {
        auto* ctx = op.getContext();
        if (op.getAxesAttr() && op.getSignalSizeAttr()) {
            return mlir::failure();
        }
        bool complexInputType = true;
        // RDFT input is real, so rank of input tensor map with parameter rank considered.
        if (mlir::isa<IE::RDFTOp>(op)) {
            complexInputType = false;
        }
        auto params = fftExtractParams(op.getLoc(), op, complexInputType);
        if (mlir::failed(params)) {
            return mlir::failure();
        }
        auto axes = params.value().axes;
        auto signalSize = params.value().signalSize;
        const auto axesAttr = getIntArrayAttr(ctx, axes);
        const auto signalSizeAttr = getIntArrayAttr(ctx, signalSize);
        rewriter.replaceOpWithNewOp<T>(op, op.getInput(), nullptr, nullptr, axesAttr, signalSizeAttr);
        return mlir::success();
    }
};

}  // namespace IE
}  // namespace vpux
