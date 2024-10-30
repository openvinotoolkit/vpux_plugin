//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>

using namespace vpux;

namespace {

//
// DynamicDequantizeConverter
//

class DynamicDequantizeConverter final : public mlir::OpRewritePattern<IE::DynamicDequantizeOp> {
public:
    DynamicDequantizeConverter(mlir::MLIRContext* ctx, vpux::Logger log)
            : mlir::OpRewritePattern<IE::DynamicDequantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DynamicDequantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DynamicDequantizeConverter::matchAndRewrite(IE::DynamicDequantizeOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("Found '{0}' Operation at '{1}'", origOp->getName(), origOp->getLoc());
    auto ctx = origOp->getContext();
    auto origElemType = origOp.getDstElemType();
    auto cvtElemType = mlir::Float16Type::get(ctx);

    auto scaleCvt = rewriter.createOrFold<IE::ConvertOp>(takeOpLoc(origOp, "_inCvt"), origOp.getScale(),
                                                         mlir::TypeAttr::get(cvtElemType));
    auto newDynamicDequantizeOp = rewriter.create<IE::DynamicDequantizeOp>(origOp->getLoc(), origOp.getInput(),
                                                                           scaleCvt, origOp.getZp(), cvtElemType);
    const auto outputCvtToOrig = rewriter.createOrFold<IE::ConvertOp>(
            takeOpLoc(origOp, "_outCvt"), newDynamicDequantizeOp.getOutput(), mlir::TypeAttr::get(origElemType));
    rewriter.replaceOp(origOp, outputCvtToOrig);
    return mlir::success();
}

//
// TopKConverter
//

class TopKConverter final : public mlir::OpRewritePattern<IE::TopKOp> {
public:
    TopKConverter(mlir::MLIRContext* ctx, vpux::Logger log): mlir::OpRewritePattern<IE::TopKOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TopKOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult TopKConverter::matchAndRewrite(IE::TopKOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found '{0}' Operation at '{1}'", origOp->getName(), origOp->getLoc());

    const auto inputElemType = origOp.getInput().getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto elemTypeFP16 = mlir::Float16Type::get(inputElemType.getContext());
    const auto inputCvtToFP16 = rewriter.createOrFold<IE::ConvertOp>(
            takeOpLoc(origOp, "cvt_in_fp16"), origOp.getInput(), mlir::TypeAttr::get(elemTypeFP16));

    mlir::IRMapping mapper;
    mapper.map(origOp.getInput(), inputCvtToFP16);
    auto newOp = rewriter.clone(*origOp, mapper);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ELEM_TYPE);

    const auto outputCvtToOrig = rewriter.createOrFold<IE::ConvertOp>(takeOpLoc(origOp, "out_cvt"), newOp->getResult(0),
                                                                      mlir::TypeAttr::get(inputElemType));
    origOp.getOutputValues().replaceAllUsesWith(outputCvtToOrig);
    origOp.getTargetShape().replaceAllUsesWith(newOp->getResult(1));
    rewriter.eraseOp(origOp);

    return mlir::success();
}

//
// AdjustSoftwareOpsPrecisionPass
//

class AdjustSoftwareOpsPrecisionPass final : public IE::AdjustSoftwareOpsPrecisionBase<AdjustSoftwareOpsPrecisionPass> {
public:
    explicit AdjustSoftwareOpsPrecisionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void AdjustSoftwareOpsPrecisionPass::safeRunOnModule() {
    auto& ctx = getContext();

    const auto isLegalTopKOp = [](IE::TopKOp op) {
        const auto inputElemType = op.getInput().getType().cast<vpux::NDTypeInterface>().getElementType();
        return inputElemType.isF16() || inputElemType.isF32() || inputElemType.isInteger(32);
    };

    const auto isLegalDynamicDequantizeOp = [](IE::DynamicDequantizeOp op) {
        const auto dstElemType = op.getDstElemType();
        return !dstElemType.isF32();
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<IE::ConvertOp>();
    target.addDynamicallyLegalOp<IE::TopKOp>(isLegalTopKOp);
    target.addDynamicallyLegalOp<IE::DynamicDequantizeOp>(isLegalDynamicDequantizeOp);
    target.markUnknownOpDynamicallyLegal([](mlir::Operation*) {
        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<TopKConverter>(&ctx, _log);
    patterns.add<DynamicDequantizeConverter>(&ctx, _log);

    auto module = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertOpsPrecisionToFP16Pass
//

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustSoftwareOpsPrecisionPass(Logger log) {
    return std::make_unique<AdjustSoftwareOpsPrecisionPass>(log);
}
