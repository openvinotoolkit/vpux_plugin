//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::DeformableConvolutionOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DeformableConvolutionOpAdaptor defConv(operands, attrs, prop);
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

    const auto inType = defConv.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = getShape(defConv.getInput()).raw();
    const auto kernelShape = getShape(defConv.getKernel()).raw();
    const auto offsetShape = getShape(defConv.getOffset()).raw();

    SmallVector<int64_t> outputShape{inShape[0],       // number of batches
                                     kernelShape[0],   // number of kernel output channels
                                     offsetShape[2],   // spatial axes Y
                                     offsetShape[3]};  // spatial axes X

    const auto outType = inType.changeShape(Shape(outputShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
