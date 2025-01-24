//
// Copyright (C) 2022-2024 Intel Corporation.
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

    const auto inDataType = lstm.getInputData().getType();
    const auto inDataShape = mlir::cast<vpux::NDTypeInterface>(inDataType).getShape();

    const auto initialHiddenStateType = mlir::cast<vpux::NDTypeInterface>(lstm.getInitialHiddenState().getType());
    const auto initialHiddenStateShape = initialHiddenStateType.getShape();
    const auto elementType = initialHiddenStateType.getElementType();

    const auto batchSize = initialHiddenStateShape[Dim(0)];
    const auto numDirections = initialHiddenStateShape[Dim(1)];
    const auto hiddenSize = initialHiddenStateShape.back();

    const auto lengthIndex = inDataShape.size() - 2;
    int64_t sequenceLength = inDataShape[Dim(lengthIndex)];

    const SmallVector<int64_t> outputHiddenValuesShape{batchSize, numDirections, sequenceLength, hiddenSize};

    if (inDataShape.isStatic()) {
        inferredReturnShapes.emplace_back(outputHiddenValuesShape, elementType);  // outputHiddenValues
    } else {
        auto outHVBounds = SmallVector<int64_t>(outputHiddenValuesShape.size());
        const auto inDataBoundedType = mlir::cast<vpux::BoundedTypeInterface>(inDataType);

        for (size_t i = 0; i < outputHiddenValuesShape.size(); i++) {
            if (outputHiddenValuesShape[i] == mlir::ShapedType::kDynamic) {
                outHVBounds[i] = parseIntArrayAttr<int64_t>(inDataBoundedType.getBounds())[lengthIndex];
            } else {
                outHVBounds[i] = outputHiddenValuesShape[i];
            }
        }
        auto outDesc = vpux::getTensorAttr(ctx, DimsOrder::fromNumDims(outputHiddenValuesShape.size()), nullptr,
                                           getIntArrayAttr(ctx, outHVBounds));
        inferredReturnShapes.emplace_back(outputHiddenValuesShape, elementType, outDesc);  // outputHiddenValues
    }

    inferredReturnShapes.emplace_back(initialHiddenStateShape, elementType);  // outputHiddenState
    inferredReturnShapes.emplace_back(initialHiddenStateShape, elementType);  // outputCellState

    return mlir::success();
}
