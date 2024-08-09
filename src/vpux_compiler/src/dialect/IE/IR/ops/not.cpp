//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::LogicalNotOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::LogicalNotOpAdaptor logicalNot(operands, attrs, prop);
    if (mlir::failed(logicalNot.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = logicalNot.getInput1().getType().cast<mlir::ShapedType>();

    inferredReturnShapes.emplace_back(in1Type.getShape(), in1Type.getElementType());

    return mlir::success();
}
