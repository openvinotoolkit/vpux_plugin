//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/convert_to_mixed_precision.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/Value.h>

using namespace vpux;
using namespace IE;

mlir::LogicalResult FloatOutConvRewriter::matchAndRewrite(IE::ConvolutionOp convolutionOp,
                                                          mlir::PatternRewriter& rewriter) const {
    if (IE::areAnyUserQuantizeOps(convolutionOp) || !_isMixPrecisionSupported(convolutionOp, false, _log)) {
        return mlir::failure();
    }
    if (mlir::failed(checkRescaledBiasRange(convolutionOp))) {
        return mlir::failure();
    }

    auto dequantizeInput = IE::findQuantizedInput(convolutionOp.getInput(), false);
    auto filterDequantizeInput = IE::findQuantizedInput(convolutionOp.getFilter(), true);

    if (dequantizeInput == nullptr || filterDequantizeInput == nullptr) {
        return mlir::failure();
    }

    auto newConv = rewriter.create<IE::ConvolutionOp>(
            convolutionOp->getLoc(), convolutionOp.getType(), dequantizeInput, filterDequantizeInput,
            convolutionOp.getBias(), convolutionOp.getStrides(), convolutionOp.getPadsBegin(),
            convolutionOp.getPadsEnd(), convolutionOp.getDilations(), convolutionOp.getPostOpAttr(),
            convolutionOp.getClampAttr(), convolutionOp.getStaticScaleAttr());
    if (!IE::checkRescaledQuantApproximationForConvBasedOp(newConv)) {
        rewriter.eraseOp(newConv);
        return mlir::failure();
    }

    rewriter.replaceOp(convolutionOp, newConv.getOutput());

    return mlir::success();
}

mlir::LogicalResult FloatOutGroupConvRewriter::matchAndRewrite(IE::GroupConvolutionOp groupConvolutionOp,
                                                               mlir::PatternRewriter& rewriter) const {
    if (IE::areAnyUserQuantizeOps(groupConvolutionOp) || !_isMixPrecisionSupported(groupConvolutionOp, false, _log)) {
        return mlir::failure();
    }
    if (mlir::failed(checkRescaledBiasRange(groupConvolutionOp))) {
        return mlir::failure();
    }

    auto dequantizeType = IE::findQuantizedInput(groupConvolutionOp.getInput(), true);
    auto filterDequantizeType = IE::findQuantizedInput(groupConvolutionOp.getFilter(), true);

    if (dequantizeType == nullptr || filterDequantizeType == nullptr) {
        return mlir::failure();
    }

    auto newGroupConv = rewriter.create<IE::GroupConvolutionOp>(
            groupConvolutionOp->getLoc(), groupConvolutionOp.getType(), dequantizeType, filterDequantizeType,
            groupConvolutionOp.getBias(), groupConvolutionOp.getStrides(), groupConvolutionOp.getPadsBegin(),
            groupConvolutionOp.getPadsEnd(), groupConvolutionOp.getDilations(), groupConvolutionOp.getGroupsAttr(),
            groupConvolutionOp.getPostOpAttr(), groupConvolutionOp.getClampAttr());

    if (!IE::checkRescaledQuantApproximationForConvBasedOp(newGroupConv)) {
        rewriter.eraseOp(newGroupConv);
        return mlir::failure();
    }

    rewriter.replaceOp(groupConvolutionOp, newGroupConv.getOutput());

    return mlir::success();
}

mlir::LogicalResult FloatOutAddRewriter::matchAndRewrite(IE::AddOp addOp, mlir::PatternRewriter& rewriter) const {
    if (IE::areAnyUserQuantizeOps(addOp) || !_isMixPrecisionSupported(addOp, false, _log)) {
        return mlir::failure();
    }
    // This transformation assumes that each input has IE::DequantizeOp producer
    auto lhsDequant = IE::findQuantizedInput(addOp.getInput1(), false);
    if (lhsDequant == nullptr) {
        return mlir::failure();
    }
    auto rhsDequant = IE::findQuantizedInput(addOp.getInput2(), false);
    if (rhsDequant == nullptr) {
        return mlir::failure();
    }

    auto lhsType = lhsDequant.getType().cast<vpux::NDTypeInterface>();
    auto lhsQuantType = lhsType.getElementType().cast<mlir::quant::UniformQuantizedType>();

    auto rhsType = rhsDequant.getType().cast<vpux::NDTypeInterface>();
    auto rhsQuantType = rhsType.getElementType().cast<mlir::quant::UniformQuantizedType>();

    // Check that the input Dequantize operands have compatible types
    if (lhsQuantType.getExpressedType() != rhsQuantType.getExpressedType() ||
        lhsQuantType.getStorageType() != rhsQuantType.getStorageType() ||
        lhsQuantType.isSigned() != rhsQuantType.isSigned()) {
        return mlir::failure();
    }

    // If target architecture does not support different scales, check that they are the same
    if (!_allowDifferentScales) {
        if (!isDoubleEqual(lhsQuantType.getScale(), rhsQuantType.getScale())) {
            return mlir::failure();
        }
    }

    rewriter.replaceOpWithNewOp<IE::AddOp>(addOp, addOp.getType(), lhsDequant, rhsDequant, addOp.getAutoBroadcast(),
                                           addOp.getPostOpAttr(), addOp.getClampAttr());

    return mlir::success();
}

mlir::LogicalResult FloatOutTransposedConvRewriter::matchAndRewrite(IE::TransposedConvolutionOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    if (IE::areAnyUserQuantizeOps(origOp) || !_isMixPrecisionSupported(origOp, false, _log)) {
        return mlir::failure();
    }
    if (mlir::failed(checkRescaledBiasRange(origOp))) {
        return mlir::failure();
    }

    auto dequantizeInput = IE::findQuantizedInput(origOp.getInput(), false);
    auto filterDequantizeInput = IE::findQuantizedInput(origOp.getFilter(), true);

    if (dequantizeInput == nullptr || filterDequantizeInput == nullptr) {
        return mlir::failure();
    }

    auto newTransposedConv = rewriter.create<IE::TransposedConvolutionOp>(
            origOp->getLoc(), origOp.getType(), dequantizeInput, filterDequantizeInput, origOp.getOutputShape(),
            origOp.getBias(), origOp.getStrides(), origOp.getPadsBegin(), origOp.getPadsEnd(), origOp.getDilations(),
            origOp.getOutputPaddingAttr(), origOp.getPostOpAttr(), origOp.getClampAttr());

    if (!IE::checkRescaledQuantApproximationForConvBasedOp(newTransposedConv)) {
        rewriter.eraseOp(newTransposedConv);
        return mlir::failure();
    }

    rewriter.replaceOp(origOp, newTransposedConv.getOutput());

    return mlir::success();
}
