//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::DequantizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::DequantizeOpAdaptor dequantize(operands, attrs, prop);
    if (mlir::failed(dequantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = dequantize.getInput().getType().cast<mlir::RankedTensorType>();
    const auto dstElemType = dequantize.getDstElemType();
    const auto outDesc = vpux::getTensorAttr(inType);

    inferredReturnShapes.emplace_back(inType.getShape(), dstElemType, outDesc);
    return mlir::success();
}
