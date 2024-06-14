//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux/compiler/utils/quantization.hpp>
#include "vpux/compiler/dialect/IE/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::QuantizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::QuantizeOpAdaptor quantize(operands, attrs);
    if (mlir::failed(quantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = quantize.getInput().getType().cast<mlir::ShapedType>();
    const auto dstElemType = quantize.getDstElemType();

    inferredReturnShapes.emplace_back(inType.getShape(), dstElemType);
    return mlir::success();
}

//
// fold
//

namespace {

mlir::quant::QuantizedType extractQuantizedType(mlir::Value operand) {
    const auto elemType = operand.getType().cast<mlir::ShapedType>().getElementType();
    const auto quantType = elemType.dyn_cast<mlir::quant::QuantizedType>();
    VPUX_THROW_UNLESS(quantType != nullptr, "Type must be quantized, but provided {0}", elemType);
    return quantType;
}

}  // namespace

mlir::OpFoldResult vpux::IE::QuantizeOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (const auto cst = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        // We can't add convertElemType transformation if content was dequantized before, see E#122631
        bool hasDequant = llvm::any_of(cst.getTransformations(), [](mlir::Attribute attr) {
            return attr.isa<Const::DequantizeAttr>();
        });
        if (hasDequant) {
            return nullptr;
        }

        const auto quantType = extractQuantizedType(getOutput());
        const auto quantStorageType = normalizeQuantStorageType(quantType);
        return cst.convertElemType(quantStorageType).quantCast(quantType);
    }

    if (auto dequantize = getInput().getDefiningOp<IE::DequantizeOp>()) {
        if (dequantize.getInput().getType() == getOutput().getType()) {
            return dequantize.getInput();
        }
    }

    return nullptr;
}
