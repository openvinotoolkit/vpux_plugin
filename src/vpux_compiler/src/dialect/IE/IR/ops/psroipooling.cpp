// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::PSROIPoolingOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::PSROIPoolingOpAdaptor psroiPooling(operands, attrs);
    if (mlir::failed(psroiPooling.verify(loc))) {
        return mlir::failure();
    }

    const auto outputDim = psroiPooling.getOutputDim();
    const auto groupSize = psroiPooling.getGroupSize();
    const auto inTypeFeatureMap = psroiPooling.getInput().getType().cast<mlir::ShapedType>();
    const auto inTypeCoord = psroiPooling.getCoords().getType().cast<mlir::ShapedType>();
    const auto inShapeCoord = inTypeCoord.getShape();

    if (outputDim <= 0) {
        return errorAt(loc, "Pooled size attribute outputDim should be positive.");
    }

    if (groupSize <= 0) {
        return errorAt(loc, "Group size attribute groupSize should be positive.");
    }

    SmallVector<int64_t> outputShape({inShapeCoord[0],  // num_rois
                                      outputDim,        // output channel number
                                      groupSize,        // pooled_w
                                      groupSize});      // pooled_h

    inferredReturnShapes.emplace_back(outputShape, inTypeFeatureMap.getElementType());
    return mlir::success();
}
