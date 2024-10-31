//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

namespace vpux {
namespace IE {

mlir::FailureOr<SmallVector<int64_t>> propagateShape(mlir::Location loc, ArrayRef<int64_t> inShape,
                                                     ArrayRef<int64_t> axes);
mlir::FailureOr<mlir::ArrayAttr> propagateBoundsAttr(mlir::MLIRContext* ctx, mlir::Location loc, mlir::Value value,
                                                     ArrayRef<int64_t> axes);

template <typename UnsqueezeType>
mlir::FailureOr<SmallVector<int64_t>> getAxes(UnsqueezeType unsqueeze, mlir::Location loc) {
    if (unsqueeze.getAxes() != nullptr && unsqueeze.getAxesValue().has_value()) {
        return errorAt(loc, "Ambiguous axes representation");
    }
    if (unsqueeze.getAxes() == nullptr && !unsqueeze.getAxesValue().has_value()) {
        return errorAt(loc, "Missed axes representation");
    }

    if (unsqueeze.getAxesValue().has_value()) {
        return parseIntArrayAttr<int64_t>(unsqueeze.getAxesValue().value());
    }

    auto axesConst = unsqueeze.getAxes().template getDefiningOp<Const::DeclareOp>();
    if (axesConst == nullptr) {
        return errorAt(loc, "Only constant axes are supported");
    }

    const auto axesContent = axesConst.getContent();
    auto axes = to_small_vector(axesContent.template getValues<int64_t>());
    std::sort(axes.begin(), axes.end());

    const auto inType = unsqueeze.getInput().getType().template cast<mlir::ShapedType>();
    const auto inRank = inType.getRank();
    const auto numAxes = checked_cast<int64_t>(axes.size());

    for (auto& axis : axes) {
        if (axis < 0) {
            axis += inRank + numAxes;
        }
    }

    return axes;
}

}  // namespace IE
}  // namespace vpux
