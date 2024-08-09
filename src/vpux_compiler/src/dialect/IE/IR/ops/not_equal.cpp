//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::NotEqualOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::NotEqualOpAdaptor notEqual(operands, attrs, prop);
    if (mlir::failed(notEqual.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = notEqual.getInput1().getType().cast<mlir::ShapedType>();
    const auto in2Type = notEqual.getInput2().getType().cast<mlir::ShapedType>();

    const auto outShapeRes =
            IE::broadcastEltwiseShape(in1Type.getShape(), in2Type.getShape(), notEqual.getAutoBroadcast(), loc);
    if (mlir::succeeded(outShapeRes)) {
        inferredReturnShapes.emplace_back(outShapeRes.value(), getBool8Type(ctx));
    }

    return outShapeRes;
}
