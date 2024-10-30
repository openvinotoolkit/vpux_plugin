//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ScatterNDUpdateOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ScatterNDUpdateOpAdaptor scatter(operands, attrs, prop);
    if (mlir::failed(scatter.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = scatter.getInput().getType().cast<mlir::RankedTensorType>();
    const auto outDesc = vpux::getTensorAttr(vpux::getOrder(inType), /*memSpace=*/nullptr, getBounds(inType));
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType(), outDesc);

    return mlir::success();
}
