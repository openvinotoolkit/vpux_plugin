//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/passes/IE2VPU/convert_IE_to_VPU_NCE.hpp"
#include "vpux/compiler/NPU37XX/conversion/passes/IE2VPU/convert_IE_to_VPU_NCE.hpp"

#include "vpux/compiler/NPU37XX/conversion.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/utils/VPU/ppe_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

//
// ConvToNCE
//

mlir::LogicalResult arch37xx::ConvToNCE::matchAndRewrite(IE::ConvolutionOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    const bool isCompressConvSupported = VPU::NCECompressConvolutionOp::isSupported(origOp, logCb,
                                                                                    /*checkLayout=*/true,
                                                                                    /*checkChannelAlignment=*/true);

    const auto filterShape = getShape(origOp.getFilter());
    auto OC = filterShape[Dims4D::Filter::OC];
    auto weightsConstValue = origOp.getFilter();
    auto rawFilterShape = getIntArrayAttr(rewriter, filterShape);
    mlir::IntegerAttr cmSpPatternAttr;
    if (isCompressConvSupported) {
        auto weightsConstOp = weightsConstValue.getDefiningOp<Const::DeclareOp>();
        auto weightsContentAttr = weightsConstOp.getContentAttr();
        auto origChannelVal =
                weightsContentAttr.getBaseContent().getType().cast<NDTypeInterface>().getShape()[Dims4D::Filter::IC];
        for (auto attr : weightsContentAttr.getTransformations()) {
            if (auto padWithZeroAttr = attr.dyn_cast_or_null<Const::PadWithZeroAttr>()) {
                const auto padZeroAttrPadsBegin = ShapeRef(parseIntArrayAttr<int64_t>(padWithZeroAttr.getPadBefore()));
                origChannelVal += padZeroAttrPadsBegin[Dims4D::Filter::IC];
            }
        }

        const auto outputChannels = origOp.getOutput().getType().cast<NDTypeInterface>().getShape()[Dims4D::Act::C];
        const auto origShape = Shape(
                {outputChannels, origChannelVal, filterShape[Dims4D::Filter::KY], filterShape[Dims4D::Filter::KX]});
        if (origShape[Dims4D::Filter::IC] != filterShape[Dims4D::Filter::IC]) {
            const auto currentOffset = SmallVector<int64_t>{0, 0, 0, 0};
            auto newContentAttr = weightsConstOp.getContentAttr().subview(Shape(currentOffset), origShape);
            auto newConstType = weightsConstOp.getType().cast<NDTypeInterface>().changeShape(origShape);
            auto newWeightsConstOp =
                    rewriter.create<Const::DeclareOp>(weightsConstOp.getLoc(), newConstType, newContentAttr);
            weightsConstValue = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(newWeightsConstOp.getOutput());
            weightsConstOp.replaceAllUsesWith(newWeightsConstOp.getOperation());
        }
        rawFilterShape = getIntArrayAttr(rewriter, origShape);
        const int64_t cmSpPattern = (static_cast<int64_t>(1) << origChannelVal) - 1;
        cmSpPatternAttr = getIntAttr(origOp->getContext(), cmSpPattern);
    }

    auto alignedFilter =
            VPU::alignConvWeightsTensor(rewriter, origOp->getLoc(), weightsConstValue, /*isCMajorConv=*/false);

    // Generate weights table
    Const::ContentAttr bias;
    if (origOp.getBias() != nullptr) {
        auto biasConstOp = origOp.getBias().getDefiningOp<Const::DeclareOp>();
        bias = biasConstOp.getContentAttr();
    }
    const auto ppeTaskAttr = VPU::getPPETaskAttrFromPostOpsParams(
            origOp.getInput(), origOp.getOutput(), origOp.getPostOpAttr(), origOp.getLoc(), origOp.getContext(), _arch);
    const auto weightsTableVec =
            VPU::createWeightsTableData(origOp.getInput(), origOp.getOutput(), alignedFilter, bias, OC, ppeTaskAttr,
                                        _arch, origOp.getPostOpAttr(), origOp.getStaticScaleAttr());
    const auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd()));

    if (isCompressConvSupported) {
        rewriter.replaceOpWithNewOp<VPU::NCECompressConvolutionOp>(
                origOp, origOp.getType(), origOp.getInput(), alignedFilter, weightsTable, origOp.getStridesAttr(),
                padAttr, ppeTaskAttr, rawFilterShape,
                /*multi_cluster_strategyAttr=*/nullptr, cmSpPatternAttr);
    } else {
        rewriter.replaceOpWithNewOp<VPU::NCEConvolutionOp>(
                origOp, origOp.getType(), origOp.getInput(), alignedFilter, weightsTable,
                /*activationWindow=*/nullptr, /*instructionListTable=*/nullptr, origOp.getStridesAttr(), padAttr,
                ppeTaskAttr, rawFilterShape,
                /*activation_window_channel_length=*/nullptr, /*multi_cluster_strategyAttr=*/nullptr);
    };

    return mlir::success();
}

//
// DepthConvToNCE
//

mlir::LogicalResult arch37xx::DepthConvToNCE::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // Get dimensions
    const auto filter = origOp.getFilter();
    const auto filterShape = getShape(filter);
    const auto OC = filterShape[Dims4D::Filter::OC];

    Const::ContentAttr bias;
    if (origOp.getBias() != nullptr) {
        auto biasConstOp = origOp.getBias().getDefiningOp<Const::DeclareOp>();
        bias = biasConstOp.getContentAttr();
    }

    const auto alignedFilter = VPU::alignDepthWiseWeightsTensor(rewriter, origOp.getLoc(), filter);

    // Generate weights table
    auto ppeTaskAttr = VPU::getPPETaskAttrFromPostOpsParams(
            origOp.getInput(), origOp.getOutput(), origOp.getPostOpAttr(), origOp.getLoc(), origOp.getContext(), _arch);
    auto weightsTableVec = VPU::createWeightsTableData(origOp.getInput(), origOp.getOutput(), alignedFilter, bias, OC,
                                                       ppeTaskAttr, _arch, origOp.getPostOpAttr(), nullptr);
    auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd()));
    const auto rawFilterShape = getIntArrayAttr(rewriter, filterShape);

    auto nceOp = rewriter.create<VPU::NCEDepthConvolutionOp>(
            origOp->getLoc(), origOp.getType(), origOp.getInput(), alignedFilter, weightsTable,
            /*activationWindow=*/nullptr, /*instructionListTable=*/nullptr, origOp.getStridesAttr(), padAttr,
            ppeTaskAttr, rawFilterShape,
            /*activation_window_channel_length=*/nullptr, /*multi_cluster_strategyAttr=*/nullptr);

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

//
// MaxPoolToNCE
//

mlir::LogicalResult arch37xx::MaxPoolToNCE::matchAndRewrite(IE::MaxPoolOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // Generate weights table
    auto ppeTaskAttr = VPU::getPPETaskAttrFromPostOpsParams(
            origOp.getInput(), origOp.getOutput(), origOp.getPostOpAttr(), origOp.getLoc(), origOp.getContext(), _arch);

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd()));

    auto nceOp = rewriter.create<VPU::NCEMaxPoolOp>(
            origOp->getLoc(), origOp.getType(), origOp.getInput(), /*weightsTable=*/nullptr,
            /*activationWindow=*/nullptr, origOp.getKernelSizeAttr(), origOp.getStridesAttr(), padAttr, ppeTaskAttr,
            /*activation_window_channel_length=*/nullptr,
            /*multi_cluster_strategyAttr=*/nullptr);

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

//
// AveragePoolToNCE
//

mlir::LogicalResult arch37xx::AveragePoolToNCE::matchAndRewrite(IE::AvgPoolOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto ppeTaskAttr = VPU::getNCEAveragePoolPPETaskAttr(origOp.getInput().getType(), origOp.getKernelSizeAttr(),
                                                         origOp.getOutput().getType(), origOp.getPostOpAttr(),
                                                         origOp.getLoc(), origOp.getContext(), _arch);

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd()));

    auto nceOp = rewriter.create<VPU::NCEAveragePoolOp>(origOp->getLoc(), origOp.getType(), origOp.getInput(),
                                                        origOp.getKernelSizeAttr(), origOp.getStridesAttr(), padAttr,
                                                        ppeTaskAttr, /*multi_cluster_strategyAttr=*/nullptr);

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

//
// PermuteQuantizeToNCEPermute
//

mlir::LogicalResult arch37xx::PermuteQuantizeToNCEPermute::matchAndRewrite(IE::PermuteQuantizeOp origOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), origOp->getName(), origOp->getLoc());

    auto outType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto expandedChannels = outType.getShape()[Dims4D::Act::C];
    const auto dstElemAttr = mlir::TypeAttr::get(outType.getElementType());

    auto nceOp = rewriter.create<VPU::NCEPermuteOp>(origOp->getLoc(), outType, origOp.getInput(),
                                                    getIntAttr(getContext(), expandedChannels), dstElemAttr,
                                                    origOp.getDstOrderAttr(),
                                                    /*ppeAttr=*/nullptr,
                                                    /*multi_cluster_strategyAttr=*/nullptr);

    rewriter.replaceOp(origOp, nceOp.getOutput());

    return mlir::success();
}

namespace {

//
// ConvertIEToVPUNCEPass
//

class ConvertIEToVPUNCEPass final : public arch37xx::ConvertIEToVPUNCEBase<ConvertIEToVPUNCEPass> {
public:
    explicit ConvertIEToVPUNCEPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertIEToVPUNCEPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    return mlir::success();
}

void ConvertIEToVPUNCEPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);

    mlir::ConversionTarget target(ctx);

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<vpux::IE::IEDialect>();
    target.addLegalDialect<vpux::VPU::VPUDialect>();
    target.addDynamicallyLegalOp<IE::ConvolutionOp>([&](IE::ConvolutionOp op) {
        return !VPU::NCEConvolutionOp::isSupported(op, logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true) &&
               !VPU::NCECompressConvolutionOp::isSupported(op, logCb, /*checkLayout=*/true,
                                                           /*checkChannelAlignment=*/true);
    });
    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>([&](IE::GroupConvolutionOp op) {
        return !VPU::NCEDepthConvolutionOp::isSupported(op, logCb, /*checkLayout=*/true,
                                                        /*checkChannelAlignment=*/true);
    });
    target.addDynamicallyLegalOp<IE::MaxPoolOp>([&](IE::MaxPoolOp op) {
        return !VPU::NCEMaxPoolOp::isSupported(op, logCb, /*checkLayout=*/true,
                                               /*checkChannelAlignment=*/true);
    });
    target.addDynamicallyLegalOp<IE::AvgPoolOp>([&](IE::AvgPoolOp op) {
        return !VPU::NCEAveragePoolOp::isSupported(op, logCb, /*checkLayout=*/true,
                                                   /*checkChannelAlignment=*/true);
    });

    target.addDynamicallyLegalOp<IE::PermuteQuantizeOp>([&](IE::PermuteQuantizeOp op) {
        return !VPU::NCEPermuteOp::isSupported(op, logCb, /*checkLayout=*/true,
                                               /*checkChannelAlignment=*/true);
    });

    target.addDynamicallyLegalOp<IE::AddOp>([&](IE::AddOp op) {
        const bool allowDifferentScales = true;
        const bool allowDifferentZp = true;

        return !VPU::NCEEltwiseOp::isSupported(op, allowDifferentScales, allowDifferentZp, logCb, /*checkLayout=*/true,
                                               /*checkChannelAlignment=*/true);
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<arch37xx::ConvToNCE>(&ctx, arch, _log);
    patterns.add<arch37xx::DepthConvToNCE>(&ctx, arch, _log);
    patterns.add<arch37xx::MaxPoolToNCE>(&ctx, arch, _log);
    patterns.add<arch37xx::AveragePoolToNCE>(&ctx, arch, _log);
    patterns.add<arch37xx::PermuteQuantizeToNCEPermute>(&ctx, _log);

    patterns.add<EltwiseToNCE<IE::AddOp>>(&ctx, VPU::EltwiseType::ADD, arch, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}
}  // namespace
//
// createConvertIEToVPUNCENCEPass
//

std::unique_ptr<mlir::Pass> vpux::arch37xx::createConvertIEToVPUNCEPass(Logger log) {
    return std::make_unique<ConvertIEToVPUNCEPass>(log);
}
