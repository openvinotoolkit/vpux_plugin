//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/cast_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::QuantizeCastOp::verify() {
    const auto dstElemType = getDstElemType();
    const auto inputType = getInput().getType().cast<vpux::NDTypeInterface>().getElementType();

    return vpux::isQuantizeCastValid(getLoc(), inputType, dstElemType);
}

mlir::LogicalResult vpux::IE::QuantizeCastOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::QuantizeCastOpAdaptor quantizeCast(operands, attrs, prop);
    if (mlir::failed(quantizeCast.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = quantizeCast.getInput().getType().cast<mlir::RankedTensorType>();
    const auto dstElemType = quantizeCast.getDstElemType();
    const auto outDesc = vpux::getTensorAttr(inType);

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

    inferredReturnShapes.emplace_back(inType.getShape(), dstElemType, outDesc);
    return mlir::success();
}

mlir::OpFoldResult vpux::IE::QuantizeCastOp::fold(FoldAdaptor adaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    } else if (const auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(adaptor.getInput())) {
        auto elemType = getDstElemTypeAttr().getValue();
        return attr.transform().castElemType(elemType).get();
    }

    return nullptr;
}

//
// FuseQuantizeCasts
//

namespace {

class FuseQuantizeCasts final : public mlir::OpRewritePattern<IE::QuantizeCastOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(IE::QuantizeCastOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseQuantizeCasts::matchAndRewrite(IE::QuantizeCastOp origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    // Transform
    // Input type1 -> IE.QuantizeCast type2 -> IE.QuantizeCast type3 -> Output type3
    // into
    // Input type1 -> IE.QuantizeCast type3 -> Output type3
    auto producerOp = origOp.getInput().getDefiningOp<IE::QuantizeCastOp>();
    if (producerOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::QuantizeCastOp>(origOp, origOp.getOutput().getType(), producerOp.getInput(),
                                                    origOp.getDstElemType());

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void IE::QuantizeCastOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<FuseQuantizeCasts>(ctx);
}
