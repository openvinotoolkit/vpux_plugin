//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::RegionYoloOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::RegionYoloOpAdaptor regionYolo(operands, attrs, prop);
    if (mlir::failed(regionYolo.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = regionYolo.getInput().getType().cast<mlir::ShapedType>();

    SmallVector<int64_t> outputShape;
    if (regionYolo.getDoSoftmax()) {
        for (int64_t i = 0; i < regionYolo.getAxis(); i++) {
            outputShape.push_back(inType.getShape()[i]);
        }

        size_t flat_dim = 1;
        for (int64_t i = regionYolo.getAxis(); i < regionYolo.getEndAxis() + 1; i++) {
            flat_dim *= inType.getShape()[i];
        }
        outputShape.push_back(flat_dim);

        for (size_t i = regionYolo.getEndAxis() + 1; i < inType.getShape().size(); i++) {
            outputShape.push_back(inType.getShape()[i]);
        }
    } else {
        outputShape.push_back(inType.getShape()[0]);
        outputShape.push_back((regionYolo.getClasses() + regionYolo.getCoords() + 1) *
                              checked_cast<int64_t>(regionYolo.getMask().size()));
        outputShape.push_back(inType.getShape()[2]);
        outputShape.push_back(inType.getShape()[3]);
    }

    inferredReturnShapes.emplace_back(outputShape, inType.getElementType());
    return mlir::success();
}
