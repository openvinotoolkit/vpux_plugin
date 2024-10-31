//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/utils/cast_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::QuantizeCastOp::verify() {
    const auto dstElemType = getDstElemType();
    const auto inputType = getInput().getType().cast<vpux::NDTypeInterface>().getElementType();

    return vpux::isQuantizeCastValid(getLoc(), inputType, dstElemType);
}

mlir::LogicalResult vpux::VPU::QuantizeCastOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::QuantizeCastOpAdaptor quantizeCast(operands, attrs, prop);
    if (mlir::failed(quantizeCast.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = quantizeCast.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto dstElemType = quantizeCast.getDstElemType();
    unsigned int outputWidth;
    if (auto quantizedOutput = dstElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        outputWidth = quantizedOutput.getStorageTypeIntegralWidth();
    } else if (auto quantizedOutput = dstElemType.dyn_cast<mlir::IntegerType>()) {
        outputWidth = quantizedOutput.getWidth();
    } else {
        return errorAt(loc, "Unsupported output type: {0}", dstElemType);
    }

    if (auto integerInput = inType.getElementType().dyn_cast<mlir::IntegerType>()) {
        const auto inputWidth = integerInput.getWidth();
        if (inputWidth != outputWidth) {
            return errorAt(loc, "Integer input width ({0}) differs from output width ({1})", inputWidth, outputWidth);
        }
    } else if (auto quantizedInput = inType.getElementType().dyn_cast<mlir::quant::QuantizedType>()) {
        const auto inputWidth = quantizedInput.getStorageTypeIntegralWidth();
        if (inputWidth != outputWidth) {
            return errorAt(loc, "Quantized input width ({0}) differs from output width ({1})", inputWidth, outputWidth);
        }
    } else {
        return errorAt(loc, "Unsupported combination of input and output element types: {0} -> {1}",
                       inType.getElementType(), dstElemType);
    }

    const auto outType = inType.changeElemType(dstElemType);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// TilingViewLikeOpInterface
//

vpux::InputTiling vpux::VPU::QuantizeCastOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    SmallVector<TileInfo> inputTiles;
    const auto inputShape = getShape(getInput());
    VPUX_THROW_UNLESS(inputShape.size() == outputTile.shape.size(),
                      "Can't tile QuantizeCast operation at '{0}', which has operands with different rank",
                      this->getLoc());
    inputTiles.push_back(outputTile);
    return TilingInfo{inputTiles};
}

void vpux::VPU::QuantizeCastOp::adjustAttrs(const TilingInfo&, const TileInfo&, ShapeRef) {
    // Do nothing
}

//
// DistributedCastOpInterface
//

mlir::FailureOr<std::pair<mlir::Type, VPU::DistributionInfo>> vpux::VPU::QuantizeCastOp::inferCastedTypeAndDistribution(
        vpux::NDTypeInterface inType, VPU::DistributionInfo& distribution) {
    if (inType == nullptr || mlir::isa<VPU::DistributedTensorType>(inType) ||
        distribution.getDistributionMode() == DistributionMode::NONE) {
        return mlir::failure();
    }
    const auto typeComponents = TypeComponents().setMemSpace(inType.getMemSpace());
    auto returnType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType()).changeTypeComponents(typeComponents);
    return std::make_pair(mlir::cast<mlir::Type>(returnType), distribution);
}
