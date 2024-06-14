//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ReorgYoloOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ReorgYoloOpAdaptor reorgYolo(operands, attrs);
    if (mlir::failed(reorgYolo.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = reorgYolo.getInput().getType().cast<mlir::ShapedType>();

    if (reorgYolo.getStride() <= 0) {
        return errorAt(loc, "Stride should be a natural number");
    }
    if (inType.getShape()[2] % reorgYolo.getStride() != 0) {
        return errorAt(loc, "Input H should be divisible by stride.");
    }
    if (inType.getShape()[3] % reorgYolo.getStride() != 0) {
        return errorAt(loc, "Input W should be divisible by stride.");
    }
    if (inType.getShape()[1] < reorgYolo.getStride() * reorgYolo.getStride()) {
        return errorAt(loc, "Input C >= (stride*stride) is required.");
    }

    SmallVector<int64_t> outputShape{inType.getShape()[0], inType.getShape()[1]};
    for (size_t i = 2; i < inType.getShape().size(); i++) {
        outputShape.push_back(inType.getShape()[i] / reorgYolo.getStride());
        outputShape[1] *= reorgYolo.getStride();
    }

    inferredReturnShapes.emplace_back(outputShape, inType.getElementType());
    return mlir::success();
}
