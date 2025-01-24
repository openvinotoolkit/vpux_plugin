//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

// include arith utils for getValueOrCreateConstantIndexOp declaration
#include <mlir/Dialect/Arith/Utils/Utils.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::ReLUOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ReLUOpAdaptor relu(operands, attrs, prop);
    if (mlir::failed(relu.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::RankedTensorType>(relu.getInput().getType());
    const auto outDesc = vpux::getTensorAttr(inType);
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType(), outDesc);

    return mlir::success();
}

// For ReLU operation the rules are rather trivial: input and output shapes are exactly equal.
// ReLU does not modify input shape in any way.
// If output dimension is static, corresponding input dimension is also static.
// If output dimension is dynamic, corresponding input dimension is also dynamic.
// [N, C, H, ?] reifies to [N, C, H, ?], [N, ?, H, ?] reifies to [N, ?, H, ?], etc.
mlir::LogicalResult vpux::IE::ReLUOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                        mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    SmallVector<mlir::OpFoldResult> shapes;
    const auto loc = getLoc();
    const auto inputShapedType = mlir::cast<mlir::ShapedType>(getInput().getType());
    const auto outputShapedType = mlir::cast<mlir::ShapedType>(getOutput().getType());
    for (const auto& dimIdx : irange(outputShapedType.getRank())) {
        if (outputShapedType.isDynamicDim(dimIdx)) {
            // Dynamic dimension: return mlir::Value.
            mlir::OpFoldResult dimOp = builder.createOrFold<mlir::tensor::DimOp>(loc, getInput(), dimIdx);
            // mlir::getValueOrCreateConstantIndexOp dispatches the fold result
            // It either returns a mlir::Value if dimOp holds a value
            // Or it creates a new ConstantIndexOp out of an attribute if dimOp holds an attribute
            shapes.push_back(mlir::getValueOrCreateConstantIndexOp(builder, loc, dimOp));
        } else {
            // Static dimension: return mlir::IntegerAttr.
            shapes.push_back(builder.getIndexAttr(inputShapedType.getDimSize(dimIdx)));
        }
    }
    reifiedReturnShapes.emplace_back(std::move(shapes));
    return mlir::success();
}
