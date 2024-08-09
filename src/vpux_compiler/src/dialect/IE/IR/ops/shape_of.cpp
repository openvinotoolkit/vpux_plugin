//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ShapeOfOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ShapeOfOpAdaptor shapeOf(operands, attrs, prop);
    if (mlir::failed(shapeOf.verify(loc))) {
        return mlir::failure();
    }

    // Outputs:
    // 1D tensor that is equal to input tensor shape.
    // Number of elements is equal to input tensor rank.
    // Can be empty 1D tensor if input tensor is a scalar, that means 0-dimensional tensor.

    const auto inType = shapeOf.getInput().getType().cast<mlir::ShapedType>();
    const auto inRank = inType.getRank();
    const SmallVector<int64_t> outShape = {inRank};
    const auto outElemType = shapeOf.getDstElemType();
    inferredReturnShapes.emplace_back(ArrayRef(outShape), outElemType);

    return mlir::success();
}
