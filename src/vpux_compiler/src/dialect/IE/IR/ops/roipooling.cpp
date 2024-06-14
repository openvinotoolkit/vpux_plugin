//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ROIPoolingOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ROIPoolingOpAdaptor roiPooling(operands, attrs);
    if (mlir::failed(roiPooling.verify(loc))) {
        return mlir::failure();
    }

    const auto output_size = parseIntArrayAttr<int64_t>(roiPooling.getOutputSize());
    const auto inTypeFeatureMap = roiPooling.getInput().getType().cast<mlir::ShapedType>();
    const auto inShapeFeatureMap = inTypeFeatureMap.getShape();

    const auto inTypeCoord = roiPooling.getCoords().getType().cast<mlir::ShapedType>();
    const auto inShapeCoord = inTypeCoord.getShape();

    if (inShapeFeatureMap.size() != 4) {
        return errorAt(loc, "Dimension of the feature maps input should be 4. Got {0} D tensor",
                       inShapeFeatureMap.size());
    }

    if (inShapeCoord.size() != 2) {
        return errorAt(loc, "Dimension of the ROIs input with box coordinates should be 2. Got {0} D tensor",
                       inShapeCoord.size());
    }

    if (output_size.size() != 2) {
        return errorAt(loc, "Dimension of pooled size is expected to be equal to 2. Got {0}", output_size.size());
    }

    if (output_size[0] <= 0 && output_size[1] <= 0) {
        return errorAt(loc, "Pooled size attributes pooled_h and pooled_w should should be positive.");
    }

    SmallVector<int64_t> output_shape;
    output_shape.push_back(inShapeCoord[0]);
    output_shape.push_back(inShapeFeatureMap[1]);
    output_shape.push_back(output_size[0]);
    output_shape.push_back(output_size[1]);

    inferredReturnShapes.emplace_back(output_shape, inTypeFeatureMap.getElementType());
    return mlir::success();
}
