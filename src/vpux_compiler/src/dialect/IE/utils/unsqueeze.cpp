//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/unsqueeze.hpp"

#include "vpux/compiler/utils/error.hpp"

namespace vpux {
namespace IE {

mlir::FailureOr<SmallVector<int64_t>> propagateShape(mlir::Location loc, ArrayRef<int64_t> inShape,
                                                     ArrayRef<int64_t> axes) {
    SmallVector<int64_t> outShape(inShape.size() + axes.size());

    size_t inInd = 0;
    size_t axesInd = 0;
    for (auto outInd : irange(outShape.size())) {
        if (axesInd < axes.size()) {
            const auto nextAxisInd = checked_cast<size_t>(axes[axesInd]);

            if (nextAxisInd < outInd) {
                return errorAt(loc, "Axis '{0}' occurred twice", nextAxisInd);
            }

            if (nextAxisInd == outInd) {
                outShape[outInd] = 1;
                ++axesInd;
                continue;
            }
        }

        if (inInd < inShape.size()) {
            outShape[outInd] = inShape[inInd];
            ++inInd;
            continue;
        }
    }
    if (inInd != inShape.size() || axesInd != axes.size()) {
        return errorAt(loc, "Inconsistent parameters");
    }

    return outShape;
}

mlir::FailureOr<mlir::ArrayAttr> propagateBoundsAttr(mlir::MLIRContext* ctx, mlir::Location loc, mlir::Value value,
                                                     ArrayRef<int64_t> axes) {
    const auto boundsAttr = vpux::getBounds(value);
    if (boundsAttr == nullptr) {
        return mlir::ArrayAttr(nullptr);
    }

    const auto bounds = parseIntArrayAttr<int64_t>(boundsAttr);
    const auto outBounds = vpux::IE::propagateShape(loc, bounds, axes);
    if (mlir::failed(outBounds)) {
        return mlir::failure();
    }

    return getIntArrayAttr(ctx, outBounds.value());
}

}  // namespace IE
}  // namespace vpux
