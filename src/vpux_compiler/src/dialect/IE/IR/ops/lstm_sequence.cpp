//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::LSTMSequenceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::LSTMSequenceOpAdaptor lstm(operands, attrs, prop);
    if (mlir::failed(lstm.verify(loc))) {
        return mlir::failure();
    }

    const auto initialHiddenStateType = mlir::cast<vpux::NDTypeInterface>(lstm.getInitialHiddenState().getType());
    const auto initialHiddenStateShape = initialHiddenStateType.getShape();
    const auto elementType = initialHiddenStateType.getElementType();

    const auto batchSize = initialHiddenStateShape[Dim(0)];
    const auto numDirections = initialHiddenStateShape[Dim(1)];
    const auto sequenceLength = lstm.getSequenceLength();
    const auto hiddenSize = initialHiddenStateShape.back();

    const SmallVector<int64_t> outputHiddenValuesShape{batchSize, numDirections, sequenceLength, hiddenSize};

    inferredReturnShapes.emplace_back(outputHiddenValuesShape, elementType);  // outputHiddenValues
    inferredReturnShapes.emplace_back(initialHiddenStateShape, elementType);  // outputHiddenState
    inferredReturnShapes.emplace_back(initialHiddenStateShape, elementType);  // outputCellState

    return mlir::success();
}
