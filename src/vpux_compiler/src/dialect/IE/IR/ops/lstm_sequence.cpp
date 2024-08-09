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

    const auto inType = lstm.getInitialHiddenState().getType().cast<mlir::ShapedType>();
    auto outHVShape = inType.getShape().vec();
    outHVShape.insert(outHVShape.cbegin() + 2, lstm.getSequenceLength());

    inferredReturnShapes.emplace_back(outHVShape, inType.getElementType());         // outputHiddenValues
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());  // outputHiddenState
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());  // outputCellState

    return mlir::success();
}
