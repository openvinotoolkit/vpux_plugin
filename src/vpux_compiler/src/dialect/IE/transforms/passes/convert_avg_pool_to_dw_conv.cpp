//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//         NON-QUANTIZED case
//   input                  input      weights
//      │                      │         │
//      │                      │         │ defined as a kernel
//      │                      │         │ with values 1/kern_size
//      │                      │         │          ┌───┬───┬───┐
// ┌────▼────┐              ┌──▼─────────▼───┐      │1/9│1/9│1/9│
// │ AvgPool │  ======►     │ DW convolution │      ├───┼───┼───┤
// └────┬────┘              └────────┬───────┘      │1/9│1/9│1/9│
//      │                            │              ├───┼───┼───┤
//      │                            │              │1/9│1/9│1/9│
//      ▼                            ▼              └───┴───┴───┘
//
//
//            QUANTIZED case
//                                               Weights are defined as a kernel
//      input               input    weights     with ALL the values set to 1
//        │                    │         │         ┌───┬───┬───┐
// ┌──────▼──────┐          ┌──▼──┐   ┌──▼──┐      │ 1 │ 1 │ 1 │
// │FakeQuantize │          │ FQ  │   │ FQ  │      ├───┼───┼───┤
// └──────┬──────┘          └──┬──┘   └──┬──┘      │ 1 │ 1 │ 1 │
//        │                    │         │         ├───┼───┼───┤
// ┌──────▼──────┐          ┌──▼─────────▼───┐     │ 1 │ 1 │ 1 │
// │   AvgPool   │  ====►   │ DW convolution │     └───┴───┴───┘
// └──────┬──────┘          └────────┬───────┘   Next FQ layer transforms 1 to 1/kern_size
//        │                          │           values to produce average after DWconv
// ┌──────▼──────┐                ┌──▼──┐          ┌───┬───┬───┐
// │FakeQuantize │                │ FQ  │          │1/9│1/9│1/9│
// └──────┬──────┘                └──┬──┘          ├───┼───┼───┤
//        │                          │             │1/9│1/9│1/9│
//        │                          │             ├───┼───┼───┤
//        ▼                          ▼             │1/9│1/9│1/9│
//                                                 └───┴───┴───┘

//
// ConvertAvgPoolToDWConvPass
//

class ConvertAvgPoolToDWConvPass final : public IE::ConvertAvgPoolToDWConvBase<ConvertAvgPoolToDWConvPass> {
public:
    explicit ConvertAvgPoolToDWConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class AvgPoolOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// AvgPoolOpConverter
//

class ConvertAvgPoolToDWConvPass::AvgPoolOpConverter final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    AvgPoolOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isAvgPoolQuantized(IE::AvgPoolOp& origOp) const;

    Logger _log;
};

bool ConvertAvgPoolToDWConvPass::AvgPoolOpConverter::isAvgPoolQuantized(IE::AvgPoolOp& origOp) const {
    const mlir::Operation* inputOp = origOp.getInput().getDefiningOp();
    if (inputOp == nullptr) {
        _log.trace("AvgPool's input is the region argument. Assuming it is not quantized.");
        return false;
    }
    const bool isInputLayerFQ = mlir::isa<IE::FakeQuantizeOp>(inputOp);
    const auto outputLayerUsers = origOp.getOutput().getUsers();
    bool areAllUsersFQ = !outputLayerUsers.empty() && ::llvm::all_of(outputLayerUsers, [](auto user) {
        return ::mlir::isa<IE::FakeQuantizeOp>(user);
    });
    return isInputLayerFQ && areAllUsersFQ;
}

mlir::LogicalResult ConvertAvgPoolToDWConvPass::AvgPoolOpConverter::matchAndRewrite(
        IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto& ctx = origOp.getContext();
    const auto outputShape = getShape(origOp.getOutput());
    const mlir::Location location = origOp.getLoc();
    const auto OC = outputShape[Dims4D::Act::C];
    const auto kernel = parseIntArrayAttr<int64_t>(origOp.getKernelSize());
    const auto kernelY = kernel[0];
    const auto kernelX = kernel[1];
    const float weightsScaleFactor = 1.0f / static_cast<float>(kernelY * kernelX);

    const auto elemType = origOp.getInput().getType().cast<vpux::NDTypeInterface>().getElementType();
    const SmallVector<int64_t> weightShape = {OC, 1, kernelY, kernelX};
    const auto dataStorageType = mlir::RankedTensorType::get(weightShape, elemType);

    const bool isAvgPoolQuantizedVal = isAvgPoolQuantized(origOp);
    const float weightRealVal = (isAvgPoolQuantizedVal) ? 1.0f : weightsScaleFactor;
    auto dwConvFilter = VPU::declareFloatConst(rewriter, location, weightRealVal, dataStorageType);
    auto weights = dwConvFilter.getOutput();

    if (isAvgPoolQuantizedVal) {
        _log.trace("AvgPool is quantized, replacing it by DW convolution with quantized weights.");

        const auto fqArgType = mlir::RankedTensorType::get({}, elemType);

        auto fqLevelsVal = getIntAttr(ctx, 255);  // the created FQ is artificial, levels are set to 255 in correlation
                                                  // with FQ range from 0.0f to 254.0f (255 integer values)
        auto fqLowVal = VPU::declareFloatConst(rewriter, location, 0.0f, fqArgType);
        auto fqInHighVal = VPU::declareFloatConst(rewriter, location, 254.0f, fqArgType);
        auto fqOutHighVal = VPU::declareFloatConst(rewriter, location, 254.0f * weightsScaleFactor, fqArgType);

        IE::FakeQuantizeOp inputLayerFQ = origOp.getInput().getDefiningOp<IE::FakeQuantizeOp>();

        IE::FakeQuantizeOp quantizationForWeights = rewriter.create<IE::FakeQuantizeOp>(
                origOp.getLoc(), dataStorageType, dwConvFilter.getOutput(), fqLowVal, fqInHighVal, fqLowVal,
                fqOutHighVal, fqLevelsVal, inputLayerFQ.getLowFpTypeAttr(), inputLayerFQ.getAutoBroadcastAttr());
        weights = quantizationForWeights.getOutput();
    }

    const SmallVector<int32_t> dilations = {1, 1};
    auto dilationsAttr = getIntArrayAttr(ctx, dilations);

    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(origOp, origOp.getInput(), weights, /*bias=*/nullptr,
                                                        origOp.getStridesAttr(), origOp.getPadsBeginAttr(),
                                                        origOp.getPadsEndAttr(), dilationsAttr, getIntAttr(ctx, OC),
                                                        /*post_opAttr=*/nullptr, /*clampAttr=*/nullptr);

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertAvgPoolToDWConvPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto isLegal = [&](IE::AvgPoolOp origOp) {
        const auto inputShape = getShape(origOp.getInput());
        const auto inputBatch = inputShape[Dims4D::Act::N];
        // Batch unrolling is actually possible for GroupConv, but leads to a large amount of NCE tasks
        // At this point, uPA task seems more effective.
        if (inputBatch != vpux::VPU::NCEInvariant::SUPPORTED_BATCH_SIZE) {
            return true;
        }
        const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.getKernelSize());
        const auto KY = kernelSize[0];
        const auto KX = kernelSize[1];

        const auto kernelStrides = parseIntArrayAttr<int64_t>(origOp.getStrides());
        const auto SY = kernelStrides[0];
        const auto SX = kernelStrides[1];

        const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
        const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());
        const auto padTop = padsBegin[0];
        const auto padBottom = padsEnd[0];
        const auto padLeft = padsBegin[1];
        const auto padRight = padsEnd[1];

        // The logic is reversed here. If AvgPoolOp can be represented as an NCE task, it becomes illegal.
        const auto arch = VPU::getArch(origOp->getParentOfType<mlir::ModuleOp>());
        return mlir::failed(VPU::NCEInvariant::verifyKernel(origOp->getLoc(), KY, KX, SY, SX, padTop, padBottom,
                                                            padLeft, padRight, arch));
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::AvgPoolOp>(isLegal);
    target.addLegalOp<IE::FakeQuantizeOp>();
    target.addLegalOp<IE::GroupConvolutionOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AvgPoolOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertAvgPoolToDWConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertAvgPoolToDWConvPass(Logger log) {
    return std::make_unique<ConvertAvgPoolToDWConvPass>(log);
}
