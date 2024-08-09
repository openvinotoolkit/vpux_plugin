//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::LSTMSequenceOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::LSTMSequenceOpAdaptor lstm(operands, attrs, prop);
    if (mlir::failed(lstm.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = lstm.getInitialHiddenState().getType().cast<vpux::NDTypeInterface>();
    auto outHVShape = inType.getShape().raw().vec();
    outHVShape.insert(outHVShape.cbegin() + 2, lstm.getSequenceLength());

    const auto outType = inType.changeShape(Shape(outHVShape));

    inferredReturnTypes.push_back(outType);  // outputHiddenValues
    inferredReturnTypes.push_back(inType);   // outputHiddenState
    inferredReturnTypes.push_back(inType);   // outputCellState

    return mlir::success();
}
