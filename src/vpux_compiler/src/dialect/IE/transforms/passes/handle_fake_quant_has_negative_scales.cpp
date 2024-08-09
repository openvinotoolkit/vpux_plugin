//
// Copyright (C) 2023 Intel Corporation.
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
    std::tie(quantMin, quantMax, storageType) = getStorageParams(origOp.getContext(), origOp.getLevels().value(),
                                                                 Const::hasNegativeValues(inLowConst.getContent()));

    const auto outLowContent = outLowConst.getContent();
    auto outLowVals = SmallVector<float>(outLowContent.getValues<float>());
    const auto outHighContent = outHighConst.getContent();
    auto outHighVals = SmallVector<float>(outHighContent.getValues<float>());
    broadcastRange(outLowVals, outHighVals, origOp.getAutoBroadcast());
    VPUX_THROW_UNLESS(outLowVals.size() == outHighVals.size(),
                      "FakeQuantize output low size '{0}' not equal with output high size '{1}'", outLowVals.size(),
                      outHighVals.size());

    // Update Scales and ZeroPoints
    const auto outChannelSize = outLowVals.size();
    SmallVector<double> updatedScales(outChannelSize);
    SmallVector<int64_t> updatedZeroPoints(outChannelSize);
    SmallVector<bool> scalesNegativeMask(outChannelSize, false);
    SmallVector<float> newOutLowVals = outLowVals;
    SmallVector<float> newOutHighVals = outHighVals;
    loop_1d(LoopExecPolicy::Parallel, origOp.getContext(), outChannelSize, [&](size_t idx) {
        const auto outLowVal = outLowVals[idx];
        const auto outHighVal = outHighVals[idx];
        if (outLowVal > 0 && outHighVal < 0) {
            scalesNegativeMask[idx] = true;
            newOutLowVals[idx] *= -1;
            newOutHighVals[idx] *= -1;
        }
        std::tie(updatedScales[idx], updatedZeroPoints[idx]) =
                calcScaleAndZeroPoint(quantMin, quantMax, newOutLowVals[idx], newOutHighVals[idx]);
    });

    // Update Quantized Const Values: Q' = (S < 0) ? 2 * ZP - Q : Q
    auto quantWeights = origOp.getInput().getDefiningOp<Const::DeclareOp>();
    const auto weightsContent = quantWeights.getContent();
    const auto weightsVal = weightsContent.getValues<float>();
    VPUX_THROW_UNLESS(weightsVal.size() % updatedScales.size() == 0,
                      "Got unexpected weights size '{0}' and scales size '{1}'", weightsVal.size(),
                      updatedScales.size());
    auto kernelSize = checked_cast<size_t>(weightsVal.size() / updatedScales.size());

    SmallVector<float> newWeightsVal(weightsVal.size());
    bool isUpdatedWeightsOutOfRange = false;
    loop_2d(LoopExecPolicy::Parallel, origOp.getContext(), updatedScales.size(), kernelSize,
            [&](int64_t scalesIdx, int64_t kernelIdx) {
                const auto weightsIdx = scalesIdx * kernelSize + kernelIdx;
                const auto origVal = weightsVal[weightsIdx];
                const auto newVal = 2 * updatedZeroPoints[scalesIdx] - origVal;
                newWeightsVal[weightsIdx] = scalesNegativeMask[scalesIdx] ? newVal : origVal;

                if (scalesNegativeMask[scalesIdx] && (newVal < quantMin || newVal > quantMax)) {
                    innerLog.trace("New weights '{0}' out of rang ['{1}', '{2}']", newVal, quantMin, quantMax);
                    isUpdatedWeightsOutOfRange = true;
                }
            });

    if (isUpdatedWeightsOutOfRange) {
        return mlir::failure();
    }

    // Update constant data
    auto newQuantWeights = Const::updateConstStorageValues(innerLog, rewriter, quantWeights, newWeightsVal);
    auto newOutLowConst = Const::updateConstStorageValues(innerLog, rewriter, outLowConst, newOutLowVals);
    auto newOutHighConst = Const::updateConstStorageValues(innerLog, rewriter, outHighConst, newOutHighVals);
    if (mlir::failed(newQuantWeights) || mlir::failed(newOutLowConst) || mlir::failed(newOutHighConst)) {
        innerLog.trace("Cannot update constant storage values");
        return mlir::failure();
    }

    // Update FakeQuantize output parameters
    rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(
            origOp, newQuantWeights.value(), origOp.getInputLow(), origOp.getInputHigh(), newOutLowConst.value(),
            newOutHighConst.value(), origOp.getLevelsAttr(), origOp.getLowFpTypeAttr(), origOp.getAutoBroadcastAttr());

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

        if (!origOp.getLevels().has_value()) {
            _log.trace("Levels attribute has no value.");
            return true;
        }

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
        broadcastRange(outLowVals, outHighVals, origOp.getAutoBroadcast());
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
