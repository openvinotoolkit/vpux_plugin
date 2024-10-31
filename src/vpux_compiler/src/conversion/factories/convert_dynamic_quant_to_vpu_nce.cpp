//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/Support/LogicalResult.h>
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/utils/VPU/ppe_utils.hpp"

using namespace vpux;

namespace {

//
// DynamicQuantToVPUNCE
//

class DynamicQuantToVPUNCE final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    DynamicQuantToVPUNCE(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// DynamicQuantToVPUNCE
//

mlir::LogicalResult DynamicQuantToVPUNCE::matchAndRewrite(IE::ConvolutionOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto dynamicDequant = origOp.getFilter().getDefiningOp<IE::DynamicDequantizeOp>();

    const auto filterShape = getShape(dynamicDequant->getOperand(0));
    auto weightsValue = dynamicDequant->getOperand(0);
    auto rawFilterShape = getIntArrayAttr(rewriter, filterShape);

    _log.trace("Align Conv Weights tensor for dynamic quantize case.");
    auto alignedFilter = VPU::alignConvWeightsTensor(rewriter, origOp->getLoc(), weightsValue);

    auto wtOutType = dynamicDequant.getScale().getType().cast<NDTypeInterface>();
    auto shape = wtOutType.getShape();
    wtOutType = wtOutType.changeShape(ShapeRef({shape[Dims4D::Act::C], 1, 1, 4}));
    wtOutType = wtOutType.changeElemType(getSInt32Type(rewriter.getContext()));
    auto weightsTable = rewriter.create<VPU::PopulateWeightTableOp>(dynamicDequant->getLoc(),
                                                                    dynamicDequant->getOperand(1), wtOutType, 0, 0);

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd()));
    const auto ppeOpaqueAttr = VPU::PpeVersionConfig::retrievePPEAttribute(origOp);

    rewriter.replaceOpWithNewOp<VPU::NCEConvolutionOp>(
            origOp, origOp.getType(), origOp.getInput(), alignedFilter, weightsTable->getResult(0),
            /*instructionListTable=*/nullptr, origOp.getStridesAttr(), padAttr, ppeOpaqueAttr, rawFilterShape,
            /*multi_cluster_strategyAttr=*/nullptr);

    rewriter.eraseOp(dynamicDequant);
    return mlir::success();
}

//
// ConvertDynamicQuantToVPUNCEPass
//

class ConvertDynamicQuantToVPUNCEPass final : public ConvertDynamicQuantToVPUNCEBase<ConvertDynamicQuantToVPUNCEPass> {
public:
    explicit ConvertDynamicQuantToVPUNCEPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertDynamicQuantToVPUNCEPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    mlir::ConversionTarget target(ctx);
    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<vpux::IE::IEDialect>();
    target.addLegalDialect<vpux::VPU::VPUDialect>();

    target.addDynamicallyLegalOp<IE::ConvolutionOp>([&](IE::ConvolutionOp op) {
        return !VPU::NCEConvolutionOp::isSupported(op, logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true) ||
               !mlir::isa<IE::DynamicDequantizeOp>(op.getFilter().getDefiningOp());
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DynamicQuantToVPUNCE>(&ctx, _log);
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertDynamicQuantToVPUNCEPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertDynamicQuantToVPUNCEPass(Logger log) {
    return std::make_unique<ConvertDynamicQuantToVPUNCEPass>(log);
}
