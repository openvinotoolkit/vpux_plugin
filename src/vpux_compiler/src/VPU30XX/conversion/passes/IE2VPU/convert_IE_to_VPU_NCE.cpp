//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/passes/IE2VPU/convert_IE_to_VPU_NCE.hpp"
#include "vpux/compiler/VPU30XX/conversion/passes/IE2VPU/convert_IE_to_VPU_NCE.hpp"

#include "vpux/compiler/VPU30XX/conversion.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/utils/VPU/ppe_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

//
// ConvToNCE
//

mlir::LogicalResult arch30xx::ConvToNCE::matchAndRewrite(IE::ConvolutionOp origOp,
                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    Const::ContentAttr bias;
    if (origOp.getBias() != nullptr) {
        auto biasConstOp = origOp.getBias().getDefiningOp<Const::DeclareOp>();
        bias = biasConstOp.getContentAttr();
    }

    const auto filterShape = getShape(origOp.getFilter());
    const auto IC = filterShape[Dims4D::Filter::IC];
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    // Generate activation window
    mlir::IntegerAttr activationWindowChannelLength;
    mlir::Value activationWindow = nullptr;
    bool isCMajorConv = DimsOrder::fromValue(origOp.getInput()) == DimsOrder::NCHW;

    if (isCMajorConv) {
        const auto kernelSize = Shape{KY, KX};
        const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(origOp.getStrides()));
        const auto origInputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();

        const auto bitPatternSize = VPU::NCESparsity::getBitPatternSize(VPU::NCESparsity::Mode::CM_CONV, kernelSize,
                                                                        kernelStrides[Dims4D::Strides::X],
                                                                        origInputType.getElementType(), IC);

        const auto fakeSparsity = VPU::NCESparsity::getFakeSparsity(VPU::NCESparsity::Mode::CM_CONV, kernelSize,
                                                                    kernelStrides[Dims4D::Strides::X],
                                                                    origInputType.getElementType(), IC);

        activationWindowChannelLength = getIntAttr(getContext(), bitPatternSize);
        activationWindow = VPU::createActivationWindowTensor(rewriter, origOp->getLoc(), fakeSparsity);
    }

    auto alignedFilter = VPU::alignConvWeightsTensor(rewriter, origOp->getLoc(), origOp.getFilter(), isCMajorConv);

    // Generate weights table
    const auto ppeTaskAttr = VPU::getPPETaskAttrFromPostOpsParams(
            origOp.getInput(), origOp.getOutput(), origOp.getPostOpAttr(), origOp.getLoc(), origOp.getContext(), _arch);

    VPUX_THROW_WHEN(origOp.getStaticScaleAttr() != nullptr, "Static scale is not supported on this architecture");
    const auto weightsTableVec = VPU::createWeightsTableData(origOp.getInput(), origOp.getOutput(), alignedFilter, bias,
                                                             OC, ppeTaskAttr, _arch, origOp.getPostOpAttr(), nullptr);
    const auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);

    const auto instructionListTableVec =
            VPU::createInstructionListTableData(origOp.getOutput(), origOp.getPostOpAttr(), _arch);
    const auto instructionListTable =
            VPU::createInstructionListTableTensor(rewriter, origOp->getLoc(), instructionListTableVec);

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd()));
    const auto rawFilterShape = getIntArrayAttr(rewriter, filterShape);

    auto nceOp = rewriter.create<VPU::NCEConvolutionOp>(
            origOp->getLoc(), origOp.getType(), origOp.getInput(), alignedFilter, weightsTable, activationWindow,
            instructionListTable, origOp.getStridesAttr(), padAttr, ppeTaskAttr, rawFilterShape,
            activationWindowChannelLength, /*multi_cluster_strategyAttr=*/nullptr);

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

//
// DepthConvToNCE
//

mlir::LogicalResult arch30xx::DepthConvToNCE::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // Get dimensions
    const auto filter = origOp.getFilter();
    const auto filterShape = getShape(filter);
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    // Generate activation window
    const auto origInputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto origInputShape = origInputType.getShape();
    const auto IC = origInputShape[Dims4D::Act::C];

    const auto kernelSize = Shape{KY, KX};
    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(origOp.getStrides()));
    const auto bitPatternSize =
            VPU::NCESparsity::getBitPatternSize(VPU::NCESparsity::Mode::DW_CONV, kernelSize,
                                                kernelStrides[Dims4D::Strides::X], origInputType.getElementType(), IC);
    const auto activationWindowChannelLength = getIntAttr(getContext(), bitPatternSize);

    const auto fakeSparsity =
            VPU::NCESparsity::getFakeSparsity(VPU::NCESparsity::Mode::DW_CONV, kernelSize,
                                              kernelStrides[Dims4D::Strides::X], origInputType.getElementType(), IC);
    const auto activationWindow = VPU::createActivationWindowTensor(rewriter, origOp->getLoc(), fakeSparsity);

    Const::ContentAttr bias;
    if (origOp.getBias() != nullptr) {
        auto biasConstOp = origOp.getBias().getDefiningOp<Const::DeclareOp>();
        if (biasConstOp == nullptr) {
            return matchFailed(_log, rewriter, origOp, "[{0}] '{1}' at '{2}' has non constant biases", getDebugName(),
                               origOp->getName(), origOp->getLoc());
        }

        bias = biasConstOp.getContentAttr();
    }

    const auto alignedFilter = VPU::alignDepthWiseWeightsTensor(rewriter, origOp.getLoc(), filter);

    // Generate weights table
    auto ppeTaskAttr = VPU::getPPETaskAttrFromPostOpsParams(
            origOp.getInput(), origOp.getOutput(), origOp.getPostOpAttr(), origOp.getLoc(), origOp.getContext(), _arch);
    auto weightsTableVec = VPU::createWeightsTableData(origOp.getInput(), origOp.getOutput(), alignedFilter, bias, OC,
                                                       ppeTaskAttr, _arch, origOp.getPostOpAttr(), nullptr);
    auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);

    const auto instructionListTableVec =
            VPU::createInstructionListTableData(origOp.getOutput(), origOp.getPostOpAttr(), _arch);
    const auto instructionListTable =
            VPU::createInstructionListTableTensor(rewriter, origOp->getLoc(), instructionListTableVec);

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd()));
    const auto rawFilterShape = getIntArrayAttr(rewriter, filterShape);

    auto nceOp = rewriter.create<VPU::NCEDepthConvolutionOp>(
            origOp->getLoc(), origOp.getType(), origOp.getInput(), alignedFilter, weightsTable, activationWindow,
            instructionListTable, origOp.getStridesAttr(), padAttr, ppeTaskAttr, rawFilterShape,
            activationWindowChannelLength, /*multi_cluster_strategyAttr=*/nullptr);

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

//
// MaxPoolToNCE
//

mlir::LogicalResult arch30xx::MaxPoolToNCE::matchAndRewrite(IE::MaxPoolOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    // Get dimensions
    const auto origInputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = origInputType.getShape();

    const auto IC = inputShape[Dims4D::Act::C];

    const auto kernelSize = Shape(parseIntArrayAttr<int64_t>(origOp.getKernelSize()));
    const auto kernelStrides = Shape(parseIntArrayAttr<int64_t>(origOp.getStrides()));

    const auto bitPatternSize =
            VPU::NCESparsity::getBitPatternSize(VPU::NCESparsity::Mode::POOL, kernelSize,
                                                kernelStrides[Dims4D::Strides::X], origInputType.getElementType(), IC);

    // Generate activation window
    const auto fakeSparsity =
            VPU::NCESparsity::getFakeSparsity(VPU::NCESparsity::Mode::POOL, kernelSize,
                                              kernelStrides[Dims4D::Strides::X], origInputType.getElementType(), IC);
    const auto activationWindow = VPU::createActivationWindowTensor(rewriter, origOp->getLoc(), fakeSparsity);
    const auto activationWindowChannelLength = getIntAttr(getContext(), static_cast<uint32_t>(bitPatternSize));

    // Generate weights table
    auto ppeTaskAttr = VPU::getPPETaskAttrFromPostOpsParams(
            origOp.getInput(), origOp.getOutput(), origOp.getPostOpAttr(), origOp.getLoc(), origOp.getContext(), _arch);
    auto weightsTableVec = VPU::createWeightsTableData(origOp.getInput(), origOp.getOutput(), nullptr, nullptr, IC,
                                                       ppeTaskAttr, _arch, origOp.getPostOpAttr(), nullptr);
    auto weightsTable = VPU::createWeightsTableTensor(rewriter, origOp->getLoc(), weightsTableVec);

    const auto padAttr = VPU::getPaddingAttr(getContext(), PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd()));

    auto nceOp = rewriter.create<VPU::NCEMaxPoolOp>(
            origOp->getLoc(), origOp.getType(), origOp.getInput(), weightsTable, activationWindow,
            origOp.getKernelSizeAttr(), origOp.getStridesAttr(), padAttr, ppeTaskAttr, activationWindowChannelLength,
            /*multi_cluster_strategyAttr=*/nullptr);

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

namespace {

//
// ConvertIEToVPUNCEPass
//

class ConvertIEToVPUNCEPass final : public arch30xx::ConvertIEToVPUNCEBase<ConvertIEToVPUNCEPass> {
public:
    explicit ConvertIEToVPUNCEPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

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
        return !VPU::NCEConvolutionOp::isSupported(op, logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true);
    });
    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>([&](IE::GroupConvolutionOp op) {
        return !VPU::NCEDepthConvolutionOp::isSupported(op, logCb, /*checkLayout=*/true,
                                                        /*checkChannelAlignment=*/true);
    });
    target.addDynamicallyLegalOp<IE::MaxPoolOp>([&](IE::MaxPoolOp op) {
        return !VPU::NCEMaxPoolOp::isSupported(op, logCb, /*checkLayout=*/true,
                                               /*checkChannelAlignment=*/true);
    });
    target.addDynamicallyLegalOp<IE::AddOp>([&](IE::AddOp op) {
        const bool allowDifferentScales = false;
        const bool allowDifferentZp = true;

        return !VPU::NCEEltwiseOp::isSupported(op, allowDifferentScales, allowDifferentZp, logCb, /*checkLayout=*/true,
                                               /*checkChannelAlignment=*/true);
    });
    target.addDynamicallyLegalOp<IE::SubtractOp>([&](IE::SubtractOp op) {
        const bool allowDifferentScales = false;
        const bool allowDifferentZp = true;

        return !VPU::NCEEltwiseOp::isSupported(op, allowDifferentScales, allowDifferentZp, logCb, /*checkLayout=*/true,
                                               /*checkChannelAlignment=*/true);
    });
    target.addDynamicallyLegalOp<IE::AndOp>([&](IE::AndOp op) {
        const bool allowDifferentScales = false;
        const bool allowDifferentZp = true;

        return !VPU::NCEEltwiseOp::isSupported(op, allowDifferentScales, allowDifferentZp, logCb, /*checkLayout=*/true,
                                               /*checkChannelAlignment=*/true);
    });
    target.addDynamicallyLegalOp<IE::MultiplyOp>([&](IE::MultiplyOp op) {
        const bool allowDifferentScales = true;
        const bool allowDifferentZp = true;

        return !VPU::NCEEltwiseOp::isSupported(op, allowDifferentScales, allowDifferentZp, logCb, /*checkLayout=*/true,
                                               /*checkChannelAlignment=*/true);
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<arch30xx::ConvToNCE>(&ctx, arch, _log);
    patterns.add<arch30xx::DepthConvToNCE>(&ctx, arch, _log);
    patterns.add<arch30xx::MaxPoolToNCE>(&ctx, arch, _log);
    patterns.add<EltwiseToNCE<IE::AddOp>>(&ctx, VPU::EltwiseType::ADD, arch, _log);
    patterns.add<EltwiseToNCE<IE::MultiplyOp>>(&ctx, VPU::EltwiseType::MULTIPLY, arch, _log);
    patterns.add<EltwiseToNCE<IE::SubtractOp>>(&ctx, VPU::EltwiseType::SUBTRACT, arch, _log);
    patterns.add<EltwiseToNCE<IE::AndOp>>(&ctx, VPU::EltwiseType::AND, arch, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertIEToVPUNCENCEPass
//

std::unique_ptr<mlir::Pass> vpux::arch30xx::createConvertIEToVPUNCEPass(Logger log) {
    return std::make_unique<ConvertIEToVPUNCEPass>(log);
}
