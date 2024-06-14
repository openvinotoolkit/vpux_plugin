// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::DeformablePSROIPoolingOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::DeformablePSROIPoolingOpAdaptor deformablepsroiPooling(operands, attrs);
    if (mlir::failed(deformablepsroiPooling.verify(loc))) {
        return mlir::failure();
    }

    const auto outputDim = deformablepsroiPooling.getOutputDim();
    const auto groupSize =
            deformablepsroiPooling.getGroupSize().has_value() ? deformablepsroiPooling.getGroupSize().value() : 1;
    const auto inTypeFeatureMap = deformablepsroiPooling.getInputScoreMaps().getType().cast<mlir::ShapedType>();
    const auto inTypeCoord = deformablepsroiPooling.getInputRois().getType().cast<mlir::ShapedType>();
    const auto inShapeCoord = inTypeCoord.getShape();

    if (outputDim <= 0) {
        return errorAt(loc, "Pooled size attribute outputDim should be positive.");
    }

    if (groupSize <= 0) {
        return errorAt(loc, "Group size attribute groupSize should be positive.");
    }
    if (inShapeCoord.size() != 2) {
        return errorAt(loc, "The rois input should be 2D");
    }

    if (deformablepsroiPooling.getInputTransformations() != nullptr) {
        const auto inTransformationsCoord =
                deformablepsroiPooling.getInputTransformations().getType().cast<mlir::ShapedType>();
        const auto inShapeTransCoord = inTransformationsCoord.getShape();

        if (inShapeTransCoord.size() != 4) {
            return errorAt(loc, "The transformation input should be 4D");
        }
    }

    SmallVector<int64_t> outputShape({inShapeCoord[0],  // num_rois
                                      outputDim,        // output channel number
                                      groupSize,        // pooled_w
                                      groupSize});      // pooled_h

    inferredReturnShapes.emplace_back(outputShape, inTypeFeatureMap.getElementType());
    return mlir::success();
}
