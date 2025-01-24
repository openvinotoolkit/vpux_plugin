//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/types/quantile_float/types.hpp"
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

    // Supported cast cases:
    //      quant_quantile  <--->   quant_quantile
    //      quant_quantile  <--->   quantile_float
    //      quant_uniform   <--->   quant_uniform
    //      quant_uniform   <--->   integer
    unsigned int inputWidth;
    unsigned int outputWidth;
    const auto inElemType = inType.getElementType();
    if (mlir::isa<mlir::quant::QuantileQuantizedType, mlir::quant::QuantileQuantizedPerAxisType>(inElemType)) {
        if (mlir::isa<mlir::quant::QuantileQuantizedType, mlir::quant::QuantileQuantizedPerAxisType>(dstElemType)) {
            auto quantizedOutput = dstElemType.dyn_cast<mlir::quant::QuantizedType>();
            outputWidth = quantizedOutput.getStorageTypeIntegralWidth();
        } else if (auto quantileFloatOutput = dstElemType.dyn_cast<vpux::type::QuantileFloatType>()) {
            outputWidth = quantileFloatOutput.getWidth();
        } else {
            return errorAt(loc, "Unsupported quantize cast: '{0}'->'{1}'", inElemType, dstElemType);
        }

        auto quantizedInput = inElemType.dyn_cast<mlir::quant::QuantizedType>();
        inputWidth = quantizedInput.getStorageTypeIntegralWidth();
        if (inputWidth != outputWidth) {
            return errorAt(loc, "Quantile quantized input width ({0}) differs from output width ({1})", inputWidth,
                           outputWidth);
        }
    } else if (auto quantileFloatInput = inElemType.dyn_cast<vpux::type::QuantileFloatType>()) {
        if (mlir::isa<mlir::quant::QuantileQuantizedType, mlir::quant::QuantileQuantizedPerAxisType>(dstElemType)) {
            auto quantizedOutput = dstElemType.dyn_cast<mlir::quant::QuantizedType>();
            outputWidth = quantizedOutput.getStorageTypeIntegralWidth();
        } else {
            return errorAt(loc, "Unsupported quantize cast: '{0}'->'{1}'", inElemType, dstElemType);
        }

        inputWidth = quantileFloatInput.getWidth();
        if (inputWidth != outputWidth) {
            return errorAt(loc, "Quantile float input width ({0}) differs from output width ({1})", inputWidth,
                           outputWidth);
        }
    } else if (mlir::isa<mlir::quant::UniformQuantizedType, mlir::quant::UniformQuantizedPerAxisType>(inElemType)) {
        if (mlir::isa<mlir::quant::UniformQuantizedType, mlir::quant::UniformQuantizedPerAxisType>(dstElemType)) {
            auto quantizedOutput = dstElemType.dyn_cast<mlir::quant::QuantizedType>();
            outputWidth = quantizedOutput.getStorageTypeIntegralWidth();
        } else if (auto integerOutput = dstElemType.dyn_cast<mlir::IntegerType>()) {
            outputWidth = integerOutput.getWidth();
        } else {
            return errorAt(loc, "Unsupported quantize cast: '{0}'->'{1}'", inElemType, dstElemType);
        }

        auto quantizedInput = inElemType.dyn_cast<mlir::quant::QuantizedType>();
        inputWidth = quantizedInput.getStorageTypeIntegralWidth();
        if (inputWidth != outputWidth) {
            return errorAt(loc, "Quantized input width ({0}) differs from output width ({1})", inputWidth, outputWidth);
        }
    } else if (auto integerInput = inElemType.dyn_cast<mlir::IntegerType>()) {
        if (auto quantizedOutput = dstElemType.dyn_cast<mlir::quant::QuantizedType>()) {
            outputWidth = quantizedOutput.getStorageTypeIntegralWidth();
        } else {
            return errorAt(loc, "Unsupported quantize cast: '{0}'->'{1}'", inElemType, dstElemType);
        }

        inputWidth = integerInput.getWidth();
        if (inputWidth != outputWidth) {
            return errorAt(loc, "Integer input width ({0}) differs from output width ({1})", inputWidth, outputWidth);
        }
    } else {
        return errorAt(loc, "Unsupported combination of input and output element types: {0} -> {1}", inElemType,
                       dstElemType);
    }

    const auto outType = inType.changeElemType(dstElemType);
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

mlir::OpFoldResult vpux::VPU::QuantizeCastOp::fold(FoldAdaptor adaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    } else if (const auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(adaptor.getInput())) {
        auto elemType = getDstElemTypeAttr().getValue();
        return attr.transform().castElemType(elemType).get();
    }

    return nullptr;
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
