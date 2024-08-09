//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::BitwiseNotOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::BitwiseNotOpAdaptor bitwiseNot(operands, attrs, prop);
    if (mlir::failed(bitwiseNot.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = bitwiseNot.getInput1().getType().cast<mlir::ShapedType>();

    inferredReturnShapes.emplace_back(in1Type.getShape(), in1Type.getElementType());

    return mlir::success();
}
