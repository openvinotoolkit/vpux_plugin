//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/expand_activation_channels.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

//
// AveragePoolRewriter
//
class AveragePoolRewriter final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    AveragePoolRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _log(log) {
        setDebugName("AveragePoolRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AveragePoolRewriter::matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got AveragePool layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadsEnd) -> mlir::Operation* {
        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);

        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadsEnd;

        const auto ndType = origOp.getType().cast<vpux::NDTypeInterface>();
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

        return rewriter.create<IE::AvgPoolOp>(origOp.getLoc(), newOutputType, expandedInput, origOp.getKernelSize(),
                                              origOp.getStrides(), origOp.getPadsBegin(), origOp.getPadsEnd(),
                                              origOp.getRoundingType(), origOp.getExcludePads(), origOp.getPostOpAttr(),
                                              origOp.getClampAttr());
    };

    return generalRewrite(origOp, rewriter, opCreator, extractMeaningfulOutput, _log.nest());
}

namespace {

//
// ExpandActivationChannelsPass
//

class ExpandActivationChannelsPass final :
        public IE::arch37xx::ExpandActivationChannelsBase<ExpandActivationChannelsPass> {
public:
    explicit ExpandActivationChannelsPass(const bool seOpsEnabled, const bool seExperimentalOpsEnabled, Logger log)
            : _seOpsEnabled(seOpsEnabled), _seExperimentalOpsEnabled(seExperimentalOpsEnabled) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    bool _seOpsEnabled;
    bool _seExperimentalOpsEnabled;
};

mlir::LogicalResult ExpandActivationChannelsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (seOpsEnabled.hasValue()) {
        _seOpsEnabled = seOpsEnabled.getValue();
    }
    if (seExperimentalOpsEnabled.hasValue()) {
        _seExperimentalOpsEnabled = seExperimentalOpsEnabled.getValue();
    }

    return mlir::success();
}

void ExpandActivationChannelsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    const auto isLegal = [&](mlir::Operation* op) {
        if (!_seExperimentalOpsEnabled && mlir::isa<IE::PadOp>(op)) {
            return true;
        }

        if (!_seOpsEnabled && mlir::isa<IE::SEOpInterface>(op) && !mlir::isa<IE::PadOp>(op)) {
            return true;
        }

        if (auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
            return iface.verifyChannels().succeeded();
        }

        return true;
    };

    mlir::ConversionTarget target(ctx);
    target.markUnknownOpDynamicallyLegal(isLegal);
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::ExpandOp, IE::SliceOp>();
    target.addLegalOp<IE::MultiplyOp, IE::SubtractOp, IE::AndOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<IE::MaxPoolRewriter>(&ctx, _log);
    patterns.add<AveragePoolRewriter>(&ctx, _log);
    patterns.add<IE::EltwiseRewriter<IE::AddOp>>(&ctx, _log);
    patterns.add<IE::ConvolutionRewriter>(&ctx, _log);
    patterns.add<IE::GroupConvolutionRewriter>(&ctx, _log);

    if (_seOpsEnabled) {
        patterns.add<IE::InterpolateRewriter>(&ctx, _log);
        patterns.add<IE::TransposedConvolutionRewriter>(&ctx, _log);
    }
    if (_seExperimentalOpsEnabled) {
        patterns.add<IE::PadRewriter>(&ctx, _log);
    }

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createExpandActivationChannelsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::arch37xx::createExpandActivationChannelsPass(const bool seOpsEnabled,
                                                                                   const bool seExperimentalOpsEnabled,
                                                                                   Logger log) {
    return std::make_unique<ExpandActivationChannelsPass>(seOpsEnabled, seExperimentalOpsEnabled, log);
}
