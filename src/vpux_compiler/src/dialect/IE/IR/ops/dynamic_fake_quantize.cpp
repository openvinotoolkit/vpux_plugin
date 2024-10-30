//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::DynamicFakeQuantizeOp::verify() {
    const auto levels = getLevels();
    const auto lowFpType = getLowFpType();

    if (!levels.has_value()) {
        if (!lowFpType.has_value()) {
            return errorAt(*this, "Missing both levels and low precision floating type");
        }
        if (!lowFpType->isa<mlir::Float8E4M3FNType>() && !lowFpType->isa<mlir::Float8E5M2Type>()) {
            return errorAt(*this, "Unsupported low floating point type {0}", *lowFpType);
        }
    } else {
        if (lowFpType.has_value()) {
            return errorAt(*this,
                           "Contradicting attributes, both levels and low precision floating type were provided");
        }
    }

    const auto inputShape = to_small_vector(getInput().getType().cast<mlir::ShapedType>().getShape());
    const auto scaleShape = to_small_vector(getScale().getType().cast<mlir::ShapedType>().getShape());
    if (inputShape.size() != scaleShape.size()) {
        return errorAt(*this, "Scale doesn't have same rank as input tensor.");
    }
    for (auto i : irange(scaleShape.size())) {
        if (scaleShape[i] > 1 && scaleShape[i] != inputShape[i]) {
            return errorAt(*this, "Scale dim doesn't equal input shape.");
        }
    }
    const auto zpShape = to_small_vector(getZp().getType().cast<mlir::ShapedType>().getShape());
    if (inputShape.size() != zpShape.size()) {
        return errorAt(*this, "ZeroPoint doesn't have same rank as input tensor.");
    }
    for (auto i : irange(zpShape.size())) {
        if (zpShape[i] > 1 && zpShape[i] != inputShape[i]) {
            return errorAt(*this, "ZeroPoint dim doesn't equal input shape.");
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::DynamicFakeQuantizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::DynamicFakeQuantizeOpAdaptor quantize(operands, attrs, prop);
    if (mlir::failed(quantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = quantize.getInput().getType().cast<mlir::ShapedType>();

    inferredReturnShapes.emplace_back(inputType.getShape(), inputType.getElementType());
    return mlir::success();
}
