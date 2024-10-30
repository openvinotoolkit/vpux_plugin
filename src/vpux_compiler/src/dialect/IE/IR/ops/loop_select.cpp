//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;
using namespace mlir;

mlir::LogicalResult vpux::IE::LoopSelectOp::verify() {
    if (!getDoConcatAttr()) {
        return errorAt(*this, "Attribute do_concat is required by LoopSelect op.");
    }

    if (getDoConcat()) {
        return errorAt(*this, "Concat cases is not supported for now.");
    }

    if (getDoConcat() && !getAxisAttr()) {
        return errorAt(*this, "Attribute axis is required by LoopSelect op concat cases.");
    }

    if (getDoConcat() && !getStrideAttr()) {
        return errorAt(*this, "Attribute stride is required by LoopSelect op concat cases.");
    }
    return mlir::success();
}

mlir::LogicalResult vpux::IE::LoopSelectOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));
    IE::LoopSelectOpAdaptor loopSelect(operands, attrs, prop);
    if (mlir::failed(loopSelect.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = loopSelect.getInput().getType().cast<mlir::RankedTensorType>();
    const auto inputShape = inputType.getShape();
    SmallVector<int64_t> outShape;
    for (size_t i = 0; i < inputShape.size(); ++i) {
        outShape.push_back(inputShape[i]);
    }

    const auto execCondsType = loopSelect.getExecConds().getType().cast<mlir::ShapedType>();
    const auto numIterations = execCondsType.getShape()[0];
    outShape[0] = outShape[0] / numIterations;

    const auto outDesc = vpux::getTensorAttr(inputType);
    inferredReturnShapes.emplace_back(outShape, inputType.getElementType(), outDesc);
    return mlir::success();
}
