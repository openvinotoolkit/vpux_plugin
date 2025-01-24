//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::DivideOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::DivideOpAdaptor divide(operands, attrs, prop);
    if (mlir::failed(divide.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = mlir::cast<mlir::RankedTensorType>(divide.getInput1().getType());
    const auto in2Type = mlir::cast<mlir::RankedTensorType>(divide.getInput2().getType());

    const auto outShapeRes =
            IE::broadcastEltwiseShape(in1Type.getShape(), in2Type.getShape(), divide.getAutoBroadcast(), loc);
    if (mlir::succeeded(outShapeRes)) {
        const auto outOrder =
                in1Type.getRank() >= in2Type.getRank() ? vpux::getOrder(in1Type) : vpux::getOrder(in2Type);
        const auto outDesc = getTensorAttr(outOrder, getMemorySpace(in1Type), getBounds(in1Type));
        inferredReturnShapes.emplace_back(outShapeRes.value(), in1Type.getElementType(), outDesc);
    }

    return outShapeRes;
}
