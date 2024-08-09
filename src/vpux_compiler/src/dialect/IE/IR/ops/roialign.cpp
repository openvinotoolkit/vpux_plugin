//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ROIAlignOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ROIAlignOpAdaptor roiAlign(operands, attrs, prop);
    if (mlir::failed(roiAlign.verify(loc))) {
        return mlir::failure();
    }

    const auto pooled_h = roiAlign.getPooledH();
    const auto pooled_w = roiAlign.getPooledW();
    const auto inTypeFeatureMap = roiAlign.getInput().getType().cast<mlir::ShapedType>();
    const auto inShapeFeatureMap = inTypeFeatureMap.getShape();

    const auto inTypeCoord = roiAlign.getCoords().getType().cast<mlir::ShapedType>();
    const auto inShapeCoord = inTypeCoord.getShape();

    if (inShapeFeatureMap.size() != 4) {
        return errorAt(loc, "Dimension of the feature maps input should be 4. Got {0} D tensor",
                       inShapeFeatureMap.size());
    }

    if (inShapeCoord.size() != 2) {
        return errorAt(loc, "Dimension of the ROIs input with box coordinates should be 2. Got {0} D tensor",
                       inShapeCoord.size());
    }

    if (pooled_h <= 0) {
        return errorAt(loc, "Pooled_h should be positive. Got {0}", pooled_h);
    }

    if (pooled_w <= 0) {
        return errorAt(loc, "Pooled_w should be positive. Got {0}", pooled_w);
    }

    SmallVector<int64_t> output_shape;
    output_shape.push_back(inShapeCoord[0]);
    output_shape.push_back(inShapeFeatureMap[1]);
    output_shape.push_back(pooled_h);
    output_shape.push_back(pooled_w);

    inferredReturnShapes.emplace_back(output_shape, inTypeFeatureMap.getElementType());
    return mlir::success();
}
