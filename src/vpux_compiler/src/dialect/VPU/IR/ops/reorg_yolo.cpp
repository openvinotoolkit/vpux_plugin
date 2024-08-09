//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ReorgYoloOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             std::optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ReorgYoloOpAdaptor reorgYolo(operands, attrs, prop);
    if (mlir::failed(reorgYolo.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = reorgYolo.getInput().getType().cast<vpux::NDTypeInterface>();

    if (reorgYolo.getStride() <= 0) {
        return errorAt(loc, "Stride should be a natural number");
    }
    if (inType.getShape().raw()[2] % reorgYolo.getStride() != 0) {
        return errorAt(loc, "Input H should be divisible by stride.");
    }
    if (inType.getShape().raw()[3] % reorgYolo.getStride() != 0) {
        return errorAt(loc, "Input W should be divisible by stride.");
    }
    if (inType.getShape().raw()[1] < reorgYolo.getStride() * reorgYolo.getStride()) {
        return errorAt(loc, "Input C >= (stride*stride) is required.");
    }

    SmallVector<int64_t> outputShape{inType.getShape().raw()[0], inType.getShape().raw()[1]};
    for (size_t i = 2; i < inType.getShape().size(); i++) {
        outputShape.push_back(inType.getShape().raw()[i] / reorgYolo.getStride());
        outputShape[1] *= reorgYolo.getStride();
    }

    const auto outType = inType.changeShape(Shape(outputShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
