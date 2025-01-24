//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <algorithm>

using namespace vpux;

namespace {

bool isLegalTensor(vpux::NDTypeInterface tensorType, int64_t symmetricalZeroPoint = 128) {
    // only handle U8 type with zero point of 128
    const auto elementType = tensorType.getElementType();
    if (const auto uniformType = elementType.dyn_cast_or_null<mlir::quant::UniformQuantizedType>()) {
        const int64_t zeroPoint = uniformType.getZeroPoint();
        if (!uniformType.isSigned() && uniformType.getStorageTypeIntegralWidth() == 8 &&
            zeroPoint == symmetricalZeroPoint) {
            return false;
        }
    } else if (const auto perAxisType = elementType.dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto zeroPoints = perAxisType.getZeroPoints();
        const auto isAllZeroPointSymmetrical = std::all_of(zeroPoints.begin(), zeroPoints.end(), [&](int n) -> bool {
            return n == symmetricalZeroPoint;
        });
        if (!perAxisType.isSigned() && perAxisType.getStorageTypeIntegralWidth() == 8 && isAllZeroPointSymmetrical) {
            return false;
        }
    }
    return true;
};

//
// ConvolutionRewriter
//

class ConvolutionRewriter final : public mlir::OpConversionPattern<IE::ConvolutionOp> {
public:
    ConvolutionRewriter(mlir::TypeConverter& typeConverter, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpConversionPattern<IE::ConvolutionOp>(typeConverter, ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, OpAdaptor newArgs,
                                        mlir::ConversionPatternRewriter& rewriter) const final;
    Const::DeclareOp replaceConstDeclare(Const::DeclareOp origOp, mlir::ConversionPatternRewriter& rewriter) const;

private:
    Logger _log;
};

Const::DeclareOp ConvolutionRewriter::replaceConstDeclare(Const::DeclareOp origOp,
                                                          mlir::ConversionPatternRewriter& rewriter) const {
    const auto outputType = origOp.getType();
    const auto origQuantType =
            outputType.cast<vpux::NDTypeInterface>().getElementType().dyn_cast<mlir::quant::QuantizedType>();

    const auto newType = typeConverter->convertType(outputType).cast<vpux::NDTypeInterface>();
    const auto newQuantType = newType.getElementType().cast<mlir::quant::QuantizedType>();

    _log.nest().trace("Convert content from '{0}' to '{1}'", origQuantType, newQuantType);

    auto newContentAttr = origOp.getContentAttr().transform().convertElemType(newQuantType).get();
    auto newConstantOp = rewriter.create<Const::DeclareOp>(origOp->getLoc(), newType, std::move(newContentAttr));
    return newConstantOp;
}

mlir::LogicalResult ConvolutionRewriter::matchAndRewrite(IE::ConvolutionOp origOp, OpAdaptor,
                                                         mlir::ConversionPatternRewriter& rewriter) const {
    _log.trace("Process Operation '{0}' at '{1}", origOp->getName(), origOp->getLoc());

    auto* typeConverter = this->getTypeConverter();
    VPUX_THROW_UNLESS(typeConverter != nullptr, "TypeConverter was not set");

    // Prior legality checks ensures all ops are defined and not null
    auto filterOp = origOp.getFilter().getDefiningOp<IE::DequantizeOp>();
    auto weightDeclareOp = filterOp.getInput().getDefiningOp<Const::DeclareOp>();
    auto newCstDeclareOp = replaceConstDeclare(weightDeclareOp, rewriter);

    auto newDequantizeOp = rewriter.create<IE::DequantizeOp>(origOp->getLoc(), newCstDeclareOp.getOutput(),
                                                             filterOp.getDstElemTypeAttr());
    rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(
            origOp, origOp.getInput(), newDequantizeOp, origOp.getBias(), origOp.getStrides(), origOp.getPadsBegin(),
            origOp.getPadsEnd(), origOp.getDilations(), origOp.getPostOpAttr(), origOp.getClampAttr(),
            origOp.getStaticScaleAttr(), origOp.getOutputChannelsAttr(), origOp.getInputChannelsAttr());
    return mlir::success();
}

//
// changeStorageTypeToI8
//

// change storage type to I8 and shift zp, min, max attributes by the value of storage type max
mlir::quant::QuantizedType changeStorageTypeToI8(mlir::quant::QuantizedType originQType) {
    if (const auto uniformType = originQType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        auto offset = (uniformType.getStorageTypeMax() + 1) / 2;
        return mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::Signed,
                                                      getSInt8Type(uniformType.getContext()),
                                                      uniformType.getExpressedType(), uniformType.getScale(),
                                                      /*zp=*/0, /*min=*/uniformType.getStorageTypeMin() - offset,
                                                      /*max=*/uniformType.getStorageTypeMax() - offset);
    } else if (const auto perAxisType = originQType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        const auto zeroPoints = perAxisType.getZeroPoints();

        const SmallVector<int64_t> newZeroPoints(zeroPoints.size(), 0);
        auto offset = (perAxisType.getStorageTypeMax() + 1) / 2;
        return mlir::quant::UniformQuantizedPerAxisType::get(
                mlir::quant::QuantizationFlags::Signed, getSInt8Type(perAxisType.getContext()),
                perAxisType.getExpressedType(), perAxisType.getScales(), newZeroPoints,
                perAxisType.getQuantizedDimension(), /*min=*/perAxisType.getStorageTypeMin() - offset,
                /*max=*/perAxisType.getStorageTypeMax() - offset);
    }

    VPUX_THROW("Unsupported Quantized Type '{0}'", originQType);
}

//
// ConvertWeightsToI8Pass
//

class ConvertWeightsToI8Pass final : public IE::arch37xx::ConvertWeightsToI8Base<ConvertWeightsToI8Pass> {
public:
    explicit ConvertWeightsToI8Pass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertWeightsToI8Pass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    mlir::TypeConverter typeConverter;

    typeConverter.addConversion([&](vpux::NDTypeInterface tensor) {
        auto quantType = tensor.getElementType().dyn_cast_or_null<mlir::quant::QuantizedType>();
        if (!isLegalTensor(tensor)) {
            const auto newElemType = changeStorageTypeToI8(quantType);
            return tensor.changeElemType(newElemType);
        }

        return tensor;
    });
    typeConverter.addSourceMaterialization(dummyConverter<mlir::RankedTensorType>);
    typeConverter.addTargetMaterialization(dummyConverter<mlir::RankedTensorType>);
    typeConverter.addArgumentMaterialization(dummyConverter<mlir::RankedTensorType>);

    mlir::ConversionTarget target(ctx);

    // We can't convert any operations that have operands with symmetric and asymmetric zero points, i.e.: IE::Add
    target.addDynamicallyLegalOp<IE::ConvolutionOp>([&](IE::ConvolutionOp op) {
        auto inputType = op.getInput().getType().cast<vpux::NDTypeInterface>().getElementType();
        // Input should be F16 and should not be FQ
        if (!inputType.isF16()) {
            return true;
        }
        auto inputOp = op.getInput().getDefiningOp();
        if (inputOp != nullptr && mlir::isa<IE::FakeQuantizeOp, IE::DequantizeOp>(inputOp)) {
            return true;
        }
        auto filterOp = op.getFilter().getDefiningOp<IE::DequantizeOp>();
        if (filterOp == nullptr) {
            return true;
        }
        auto weightDeclareOp = filterOp.getInput().getDefiningOp<Const::DeclareOp>();
        if (weightDeclareOp == nullptr) {
            return true;
        }

        return isLegalTensor(weightDeclareOp.getOutput().getType().cast<vpux::NDTypeInterface>());
    });
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::DequantizeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvolutionRewriter>(typeConverter, &ctx, _log);

    if (mlir::failed(applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}
}  // namespace

//
// createConvertWeightsToI8Pass
//

std::unique_ptr<mlir::Pass> vpux::IE::arch37xx::createConvertWeightsToI8Pass(Logger log) {
    return std::make_unique<ConvertWeightsToI8Pass>(log);
}
