// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::DeformableConvolutionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::DeformableConvolutionOpAdaptor defConv(operands, attrs, prop);
    if (mlir::failed(defConv.verify(loc))) {
        return mlir::failure();
    }

    const auto maskInput = defConv.getMask();
    if (maskInput == nullptr) {
        return errorAt(loc, "The case without input mask is not supported");
    }

    const auto group = defConv.getGroup();
    if (group != 1) {
        return errorAt(loc, "Group attribute should have the value 1");
    }

    const auto deformableGroup = defConv.getDeformableGroup();
    if (deformableGroup != 1) {
        return errorAt(loc, "DeformableGroup attribute should have the value 1");
    }

    const auto inType = defConv.getInput().getType().cast<mlir::ShapedType>();
    const auto inShape = getShape(defConv.getInput()).raw();
    const auto kernelShape = getShape(defConv.getKernel()).raw();
    const auto offsetShape = getShape(defConv.getOffset()).raw();

    SmallVector<int64_t> outputShape{inShape[0],       // number of batches
                                     kernelShape[0],   // number of kernel output channels
                                     offsetShape[2],   // spatial axes Y
                                     offsetShape[3]};  // spatial axes X

    inferredReturnShapes.emplace_back(outputShape, inType.getElementType());

    return mlir::success();
}
