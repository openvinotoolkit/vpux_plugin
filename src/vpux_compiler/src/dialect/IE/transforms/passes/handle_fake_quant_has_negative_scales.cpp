//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/content.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

mlir::Value updateConstStorageValues(mlir::OpBuilder& builder, Const::DeclareOp origConst, ArrayRef<float> values) {
    // Note: do a little trick here: when original constant's base content is
    // FP32, store new values also in FP32 (to ensure we do not lose any
    // information due to FP32 -> FP16 conversion). otherwise, store the data in
    // the output format (originals are either FP or some integer values that
    // were historically converted to FP16).
    const auto finalType = mlir::cast<NDTypeInterface>(origConst.getOutput().getType());
    const auto baseContentType = mlir::cast<NDTypeInterface>(origConst.getContentAttr().getBaseContent().getType());
    if (baseContentType.getElementType().isF32()) {
        const auto fp32TensorType =
                mlir::RankedTensorType::get(finalType.getShape(), mlir::Float32Type::get(builder.getContext()));
        // Note: if createFloatConst() is changed to always store data in FP32
        // (with optional conversion to FP16), then this branch won't be needed.
        return Const::createConst(builder, origConst->getLoc(), fp32TensorType, values,
                                  [finalElemType = finalType.getElementType()](Const::ContentSetup& setup) {
                                      if (bool constTypeIsAlreadyCorrect = finalElemType.isF32();
                                          constTypeIsAlreadyCorrect) {
                                          return std::move(setup);
                                      }
                                      return setup.castElemType(finalElemType);
                                  });
    }

    return Const::createFloatConst(builder, origConst->getLoc(), mlir::cast<mlir::RankedTensorType>(finalType), values);
}

//
// HandleConstWeightsFakeQuant
//

class HandleConstWeightsFakeQuant final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    HandleConstWeightsFakeQuant(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
        setDebugName("HandleConstWeightsFakeQuant");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult HandleConstWeightsFakeQuant::matchAndRewrite(IE::FakeQuantizeOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    auto innerLog = _log.nest();

    auto inLowConst = origOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = origOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = origOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();

    mlir::Type storageType;
    int64_t quantMin = 0;
    int64_t quantMax = 0;
    if (origOp.getLevels().has_value()) {
        std::tie(quantMin, quantMax, storageType) = getStorageParams(origOp.getContext(), *origOp.getLevels(),
                                                                     Const::hasNegativeValues(inLowConst.getContent()));
    } else {
        std::tie(quantMin, quantMax, storageType) = getStorageParams(origOp.getContext(), *origOp.getLowFpType());
    }

    const auto outLowContent = outLowConst.getContent();
    auto outLowVals = to_small_vector(outLowContent.getValues<float>());
    const auto outHighContent = outHighConst.getContent();
    auto outHighVals = to_small_vector(outHighContent.getValues<float>());
    // out_low.size() == out_high.size() always holds for *weights* by
    // definition (see --weights-dequantize-to-fake-quantize).
    VPUX_THROW_UNLESS(outLowVals.size() == outHighVals.size(),
                      "FakeQuantize output low size '{0}' not equal with output high size '{1}'", outLowVals.size(),
                      outHighVals.size());

    // Update Scales and ZeroPoints
    const auto outChannelSize = outLowVals.size();
    SmallVector<double> updatedScales(outChannelSize);
    SmallVector<int64_t> updatedZeroPoints(outChannelSize);
    SmallVector<bool> scalesNegativeMask(outChannelSize, false);
    loop_1d(LoopExecPolicy::Parallel, origOp.getContext(), outChannelSize, [&](size_t idx) {
        auto& outLowVal = outLowVals[idx];
        auto& outHighVal = outHighVals[idx];
        if (outLowVal > 0 && outHighVal < 0) {
            scalesNegativeMask[idx] = true;
            outLowVal *= -1;
            outHighVal *= -1;
        }
        std::tie(updatedScales[idx], updatedZeroPoints[idx]) =
                calcScaleAndZeroPoint(quantMin, quantMax, outLowVal, outHighVal);
    });

    // Update Quantized Const Values: Q' = (S < 0) ? 2 * ZP - Q : Q
    auto quantWeights = origOp.getInput().getDefiningOp<Const::DeclareOp>();
    const auto weightsContent = quantWeights.getContent();
    auto weightsVal = to_small_vector(weightsContent.getValues<float>());
    VPUX_THROW_UNLESS(weightsVal.size() % updatedScales.size() == 0,
                      "Got unexpected weights size '{0}' and scales size '{1}'", weightsVal.size(),
                      updatedScales.size());
    auto kernelSize = checked_cast<size_t>(weightsVal.size() / updatedScales.size());

    bool isUpdatedWeightsOutOfRange = false;
    loop_2d(LoopExecPolicy::Parallel, origOp.getContext(), updatedScales.size(), kernelSize,
            [&](int64_t scalesIdx, int64_t kernelIdx) {
                const auto weightsIdx = scalesIdx * kernelSize + kernelIdx;
                auto& origVal = weightsVal[weightsIdx];
                const auto newVal = 2 * updatedZeroPoints[scalesIdx] - origVal;
                origVal = scalesNegativeMask[scalesIdx] ? newVal : origVal;

                if (scalesNegativeMask[scalesIdx] && (newVal < quantMin || newVal > quantMax)) {
                    innerLog.trace("New weights '{0}' out of rang ['{1}', '{2}']", newVal, quantMin, quantMax);
                    isUpdatedWeightsOutOfRange = true;
                }
            });

    if (isUpdatedWeightsOutOfRange) {
        return mlir::failure();
    }

    // Update constant data
    auto newQuantWeights = updateConstStorageValues(rewriter, quantWeights, weightsVal);
    auto newOutLowConst = updateConstStorageValues(rewriter, outLowConst, outLowVals);
    auto newOutHighConst = updateConstStorageValues(rewriter, outHighConst, outHighVals);

    // Update FakeQuantize output parameters
    rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(
            origOp, newQuantWeights, origOp.getInputLow(), origOp.getInputHigh(), newOutLowConst, newOutHighConst,
            origOp.getLevelsAttr(), origOp.getLowFpTypeAttr(), origOp.getAutoBroadcastAttr());

    rewriter.eraseOp(quantWeights);
    rewriter.eraseOp(outLowConst);
    rewriter.eraseOp(outHighConst);

    innerLog.trace("Handle FakeQuantize at '{0}' completed", origOp->getLoc());
    return mlir::success();
}

//
// HandleFakeQuantHasNegativeScalesPass
//

class HandleFakeQuantHasNegativeScalesPass final :
        public IE::HandleFakeQuantHasNegativeScalesBase<HandleFakeQuantHasNegativeScalesPass> {
public:
    explicit HandleFakeQuantHasNegativeScalesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void HandleFakeQuantHasNegativeScalesPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::FakeQuantizeOp>([&](IE::FakeQuantizeOp origOp) {
        _log.trace("Got FakeQuantize Operation '{1}'", origOp->getLoc());

        const auto inputType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
        if (inputType.getRank() != 4) {
            _log.nest().trace("Tensor '{0}' rank should equal 4, but got '{1}'", origOp->getLoc(), inputType.getRank());
            return true;
        }

        auto constInput = origOp.getInput().getDefiningOp<Const::DeclareOp>();
        if (constInput == nullptr) {
            _log.nest().trace("Got non constant input of FakeQuantize '{0}'", origOp->getLoc());
            return true;
        }

        auto inLowConst = origOp.getInputLow().getDefiningOp<Const::DeclareOp>();
        auto inHighConst = origOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
        auto outLowConst = origOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
        auto outHighConst = origOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();
        if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
            _log.nest().trace("Got non constant parameters of FakeQuantize '{0}'", origOp->getLoc());
            return true;
        }

        const auto outLowContent = outLowConst.getContent();
        auto outLowVals = SmallVector<float>(outLowContent.getValues<float>());
        const auto outHighContent = outHighConst.getContent();
        auto outHighVals = SmallVector<float>(outHighContent.getValues<float>());
        // out_low.size() == out_high.size() always holds for *weights* by
        // definition (see --weights-dequantize-to-fake-quantize).
        VPUX_THROW_UNLESS(outLowVals.size() == outHighVals.size(),
                          "FakeQuantize output low size '{0}' not equal with output high size '{1}'", outLowVals.size(),
                          outHighVals.size());
        const auto hasNegativeScales = llvm::any_of(zip(outLowVals, outHighVals), [](const auto& vals) {
            return std::get<0>(vals) > 0 && std::get<1>(vals) < 0;
        });

        if (!hasNegativeScales) {
            _log.nest().trace("Got non negative Scales in FakeQuantize '{0}'", origOp->getLoc());
            return true;
        }

        return false;
    });
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<HandleConstWeightsFakeQuant>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createHandleFakeQuantHasNegativeScalesPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createHandleFakeQuantHasNegativeScalesPass(Logger log) {
    return std::make_unique<HandleFakeQuantHasNegativeScalesPass>(log);
}
