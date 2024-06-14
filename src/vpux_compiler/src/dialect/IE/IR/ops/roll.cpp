//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::RollOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::RollOpAdaptor roll(operands, attrs);
    if (mlir::failed(roll.verify(loc))) {
        return mlir::failure();
    }

    const auto shiftContent = roll.getShift().getDefiningOp<Const::DeclareOp>().getContent();
    const auto inShapeShift = getShape(roll.getShift());

    if (!shiftContent.isSplat() && inShapeShift.size() == 1) {
        auto shiftData = IE::constInputToData(loc, roll.getShift());
        if (mlir::failed(shiftData)) {
            return mlir::failure();
        }

        auto axesData = IE::constInputToData(loc, roll.getAxes());
        if (mlir::failed(axesData)) {
            return mlir::failure();
        }

        auto shiftShape = shiftData.value();
        auto axesShape = axesData.value();

        if (shiftShape.size() != axesShape.size()) {
            return errorAt(loc,
                           "If shift is a 1D vector, axes must be a 1D tensor of the same size. Got shift size {0} and "
                           "axes size {1}.",
                           shiftShape.size(), axesShape.size());
        }
    }

    const auto inType = roll.getData().getType().cast<mlir::ShapedType>();
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}
