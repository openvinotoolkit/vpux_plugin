//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::GRUSequenceFirstPartOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GRUSequenceFirstPartOpAdaptor gru(operands, attrs, prop);
    if (mlir::failed(gru.verify(loc))) {
        return mlir::failure();
    }

    const auto inputDataType = gru.getInputData().getType().cast<vpux::NDTypeInterface>();
    const auto weightsType = gru.getWeights().getType().cast<vpux::NDTypeInterface>();
    const auto inputDataShape = inputDataType.getShape().raw();
    const auto weightsShape = weightsType.getShape().raw();
    const auto seqLength = gru.getSeqLength();
    SmallVector<int64_t> outputShape = {inputDataShape[0], weightsShape[0], seqLength, weightsShape[1]};
    const auto outputType = weightsType.changeShape(Shape(outputShape));

    inferredReturnShapes.push_back(outputType);

    return mlir::success();
}
