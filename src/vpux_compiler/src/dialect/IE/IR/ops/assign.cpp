//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::AssignOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::AssignOpAdaptor assign(operands, attrs);
    if (mlir::failed(assign.verify(loc))) {
        return mlir::failure();
    }

    const auto rankedInType = assign.getInput().getType().cast<mlir::RankedTensorType>();
    const auto outDesc = vpux::getTensorAttr(rankedInType);
    inferredReturnShapes.emplace_back(rankedInType.getShape(), rankedInType.getElementType(), outDesc);

    return mlir::success();
}
