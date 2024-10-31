//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::DynamicDequantizeOp::verify() {
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
    auto zp = getZp();
    if (zp != nullptr) {
        const auto zpShape = to_small_vector(mlir::cast<NDTypeInterface>(zp.getType()).getShape());
        if (inputShape.size() != zpShape.size()) {
            return errorAt(*this, "ZeroPoint doesn't have same rank as input tensor.");
        }
        for (auto i : irange(zpShape.size())) {
            if (zpShape[i] > 1 && zpShape[i] != inputShape[i]) {
                return errorAt(*this, "ZeroPoint dim doesn't equal input shape.");
            }
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::DynamicDequantizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::DynamicDequantizeOpAdaptor dynamicDequantize(operands, attrs, prop);
    if (mlir::failed(dynamicDequantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = dynamicDequantize.getInput().getType().cast<mlir::RankedTensorType>();
    const auto dstElemType = dynamicDequantize.getDstElemType();
    const auto outDesc = vpux::getTensorAttr(inType);

    inferredReturnShapes.emplace_back(inType.getShape(), dstElemType, outDesc);

    return mlir::success();
}
