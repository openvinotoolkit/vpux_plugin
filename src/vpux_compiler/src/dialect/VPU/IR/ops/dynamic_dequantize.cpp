//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::DynamicDequantizeOp::verify() {
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

mlir::LogicalResult vpux::VPU::DynamicDequantizeOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DynamicDequantizeOpAdaptor dynamicDequantize(operands, attrs, prop);
    if (mlir::failed(dynamicDequantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = dynamicDequantize.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto dstElemType = dynamicDequantize.getDstElemType();

    const auto outType = inType.changeElemType(dstElemType);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
