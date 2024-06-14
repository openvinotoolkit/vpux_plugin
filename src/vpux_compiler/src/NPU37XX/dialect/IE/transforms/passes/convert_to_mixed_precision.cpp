//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/convert_to_mixed_precision.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

//
// ConvertToMixedPrecisionPass
//

class ConvertToMixedPrecisionPass final :
        public IE::arch37xx::ConvertToMixedPrecisionBase<ConvertToMixedPrecisionPass> {
public:
    explicit ConvertToMixedPrecisionPass(const bool enableFloatInQuantWeightsMixedMode, Logger log)
            : _enableFloatInQuantWeightsMixedMode(enableFloatInQuantWeightsMixedMode) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    bool _enableFloatInQuantWeightsMixedMode;
};

mlir::LogicalResult ConvertToMixedPrecisionPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (enableFloatInQuantWeightsMixedMode.hasValue()) {
        _enableFloatInQuantWeightsMixedMode = enableFloatInQuantWeightsMixedMode.getValue();
    }

    return mlir::success();
}

class FloatOutAvgPoolRewriter final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    FloatOutAvgPoolRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp avgPoolOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FloatOutAvgPoolRewriter::matchAndRewrite(IE::AvgPoolOp avgPoolOp,
                                                             mlir::PatternRewriter& rewriter) const {
    if (IE::areAnyUserQuantizeOps(avgPoolOp) || !IE::arch37xx::isMixPrecisionSupported(avgPoolOp, false, _log)) {
        return mlir::failure();
    }
    // Although the operation could support per channel quant params because is depthwise,
    // it does not have access to weights table, which is where per channel quant params
    // are placed. Only global, per tensor quantization is supported by AVG Pool.
    auto dequantizeType = IE::findQuantizedInput(avgPoolOp.getInput(), false);
    if (dequantizeType == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::AvgPoolOp>(
            avgPoolOp, avgPoolOp.getType(), dequantizeType, avgPoolOp.getKernelSize(), avgPoolOp.getStrides(),
            avgPoolOp.getPadsBegin(), avgPoolOp.getPadsEnd(), avgPoolOp.getRoundingTypeAttr(),
            avgPoolOp.getExcludePadsAttr(), avgPoolOp.getPostOpAttr(), avgPoolOp.getClampAttr());

    return mlir::success();
}

class QuantizeWithNCERewriter final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    QuantizeWithNCERewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult QuantizeWithNCERewriter::matchAndRewrite(IE::QuantizeOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    const auto maybeNCETask = origOp.getInput().getDefiningOp();
    if (maybeNCETask == nullptr) {
        return matchFailed(rewriter, origOp, "Producer is a block argument");
    }
    if (!maybeNCETask->getResult(0).hasOneUse()) {
        return matchFailed(rewriter, origOp, "NCE task has more than one consumer");
    }
    if (mlir::isa<IE::MaxPoolOp>(maybeNCETask)) {
        return matchFailed(rewriter, origOp, "IE.MaxPool does not support fp16 input and quantized output");
    }

    const auto quantType = origOp.getOutput().getType();
    const bool isPerChannel =
            quantType.cast<vpux::NDTypeInterface>().getElementType().isa<mlir::quant::UniformQuantizedPerAxisType>();
    if (mlir::isa<IE::AddOp, IE::AvgPoolOp>(maybeNCETask) && isPerChannel) {
        return matchFailed(rewriter, origOp, "IE.AvgPool and IE.Add do not support per-channel quantized output");
    }

    // NCE tasks with float input and quant output support LeakyReLU only per-tensor quantize output.
    // One would expect that with ops ran sequential: BIAS->SCALE->PRELU, we could easily support prelu and per axis
    // quant params. But actually in HW, depending on the sign of the FP BIAS result, you either execute SCALE or PRELU.
    // So for the negative values we'd have to combine the prelu alpha parameter and the requant scale into the per
    // tensor param for prelu scale. This explains why we can't have prelu with per axis quant in fp mode
    if (!IE::arch37xx::isMixPrecisionSupported(maybeNCETask, !isPerChannel, _log)) {
        return matchFailed(rewriter, origOp, "Producer {0} is not supported", maybeNCETask->getName());
    }

    auto* newNCETask = rewriter.clone(*maybeNCETask);
    newNCETask->getResult(0).setType(quantType);
    if (!IE::checkRescaledQuantApproximationForConvBasedOp(newNCETask)) {
        rewriter.eraseOp(newNCETask);
        return mlir::failure();
    }
    rewriter.replaceOp(origOp, newNCETask->getResult(0));
    rewriter.eraseOp(maybeNCETask);

    return mlir::success();
}

template <typename ConcreteOp>
class MixedFloatInQuantWeightsRewriter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    MixedFloatInQuantWeightsRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp convOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult MixedFloatInQuantWeightsRewriter<ConcreteOp>::matchAndRewrite(
        ConcreteOp convOp, mlir::PatternRewriter& rewriter) const {
    if (!IE::arch37xx::isMixPrecisionSupported(convOp, true, _log)) {
        return mlir::failure();
    }

    const auto dequantizeType = IE::findQuantizedInput(convOp.getInput(), false);
    const auto filterDequantizeType = IE::findQuantizedInput(convOp.getFilter(), true);

    // Not fit for input weights mixed precision, other rewriters will apply
    if (dequantizeType != nullptr || filterDequantizeType == nullptr) {
        return mlir::failure();
    }

    const auto quantFilterDequantizeType = filterDequantizeType.getType()
                                                   .template cast<vpux::NDTypeInterface>()
                                                   .getElementType()
                                                   .template dyn_cast<mlir::quant::QuantizedType>();
    if (quantFilterDequantizeType == nullptr) {
        return mlir::failure();
    }

    // Only signed quant is supported for input + wt mixed precision
    if (!quantFilterDequantizeType.isSigned() || !IE::isSymmetricQuantType(quantFilterDequantizeType)) {
        return mlir::failure();
    }

    const auto hasLeakyReLUConsumer = llvm::any_of(convOp->getUsers(), [](mlir::Operation* op) {
        return mlir::isa<IE::LeakyReluOp>(op);
    });

    if (mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(quantFilterDequantizeType) &&
        (hasLeakyReLUConsumer || IE::hasLeakyReLUPostOp(convOp))) {
        return mlir::failure();
    }

    mlir::IRMapping mapper;
    mapper.map(convOp.getFilter(), filterDequantizeType);
    auto newOp = rewriter.clone(*convOp, mapper);
    if (!IE::checkRescaledQuantApproximationForConvBasedOp(newOp)) {
        rewriter.eraseOp(newOp);
        return mlir::failure();
    }
    rewriter.replaceOp(convOp, newOp->getResults());

    return mlir::success();
}

void ConvertToMixedPrecisionPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    // E#67754 - MaxPool is omitted intentionally because it generates accuracy issues.
    patterns.add<vpux::IE::FloatOutConvRewriter>(&ctx, IE::arch37xx::isMixPrecisionSupported, _log);
    patterns.add<vpux::IE::FloatOutGroupConvRewriter>(&ctx, IE::arch37xx::isMixPrecisionSupported, _log);
    patterns.add<vpux::IE::FloatOutAddRewriter>(&ctx, IE::arch37xx::isMixPrecisionSupported, true, _log);
    patterns.add<vpux::IE::FloatOutTransposedConvRewriter>(&ctx, IE::arch37xx::isMixPrecisionSupported, _log);

    patterns.add<FloatOutAvgPoolRewriter>(&ctx, _log);
    patterns.add<QuantizeWithNCERewriter>(&ctx, _log);

    // Patterns for mixed precision of float input and quant weights
    if (_enableFloatInQuantWeightsMixedMode) {
        patterns.add<MixedFloatInQuantWeightsRewriter<IE::ConvolutionOp>>(&ctx, _log);
        patterns.add<MixedFloatInQuantWeightsRewriter<IE::GroupConvolutionOp>>(&ctx, _log);
    }

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertToMixedPrecision
//

std::unique_ptr<mlir::Pass> vpux::IE::arch37xx::createConvertToMixedPrecision(
        const bool enableFloatInQuantWeightsMixedMode, Logger log) {
    return std::make_unique<ConvertToMixedPrecisionPass>(enableFloatInQuantWeightsMixedMode, log);
}
