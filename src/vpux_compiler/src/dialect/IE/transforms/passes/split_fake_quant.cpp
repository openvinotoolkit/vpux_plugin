//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/quantization.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

bool checkRange(Const::ContentAttr lowConst, Const::ContentAttr highConst, IE::AutoBroadcastType broadcast,
                bool (*predicate)(const double low, const double high)) {
    const auto lowAttr = lowConst.fold();
    const auto highAttr = highConst.fold();
    if (lowAttr.isSplat() && highAttr.isSplat()) {
        const auto low = lowAttr.getSplatValue<double>();
        const auto high = highAttr.getSplatValue<double>();
        return predicate(low, high);
    }

    const auto lowVals = lowAttr.getValues<double>();
    const auto highVals = highAttr.getValues<double>();

    SmallVector<double> lows(lowVals);
    SmallVector<double> highs(highVals);
    broadcastRange(lows, highs, broadcast);

    for (auto p : zip(lows, highs)) {
        const auto lowVal = std::get<0>(p);
        const auto highVal = std::get<1>(p);
        if (!predicate(lowVal, highVal))
            return false;
    }

    return true;
}

bool containsValueZero(Const::ContentAttr lowConst, Const::ContentAttr highConst, IE::AutoBroadcastType broadcast) {
    auto containsZero = [](const double low, const double high) {
        return low <= 0 && high >= 0;
    };

    return checkRange(lowConst, highConst, broadcast, containsZero);
}

// Ranges without value zero lead to a negative zero-point which is not supported in the DPU PPE
bool hasRangeWithoutZero(IE::FakeQuantizeOp fqOp) {
    auto inLowConst = fqOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = fqOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = fqOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = fqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();

    if (!containsValueZero(inLowConst.getContentAttr(), inHighConst.getContentAttr(), fqOp.getAutoBroadcast()) ||
        !containsValueZero(outLowConst.getContentAttr(), outHighConst.getContentAttr(), fqOp.getAutoBroadcast())) {
        return true;
    }

    return false;
}

// Scalar like [7, 7] is handled separately and zero value is not required.
// In this case ZP=0, scale=scalar.
bool isScalar(IE::FakeQuantizeOp fqOp) {
    auto inLowConst = fqOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = fqOp.getInputHigh().getDefiningOp<Const::DeclareOp>();

    auto isScalarLambda = [](const double low, const double high) {
        return std::fabs(high - low) < std::numeric_limits<double>::epsilon();
    };

    return checkRange(inLowConst.getContentAttr(), inHighConst.getContentAttr(), fqOp.getAutoBroadcast(),
                      isScalarLambda);
}

//
// UseQuantDequant
//

class UseQuantDequant final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    UseQuantDequant(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
        setDebugName("UseQuantDequant");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UseQuantDequant::matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got FakeQuantize Operation '{1}'", getDebugName(), origOp->getLoc());
    auto innerLog = _log.nest();

    if (origOp.getInput().getDefiningOp<Const::DeclareOp>() != nullptr) {
        return matchFailed(innerLog, rewriter, origOp, "Got constant input");
    }

    auto inLowConst = origOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = origOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = origOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = origOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();

    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        return matchFailed(innerLog, rewriter, origOp, "Got non constant parameters");
    }

    innerLog.trace("Try to use Quantize/[QuantizeCast]/Dequantize operations");

    const auto realType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto realElemType = realType.getElementType().cast<mlir::FloatType>();

    const auto inQuantizeElemType = getQuantizedType(
            inLowConst.getContentAttr(), inHighConst.getContentAttr(), origOp.getLevels(), origOp.getLowFpType(),
            realElemType, false, origOp.getLoc(), origOp.getAutoBroadcast(), /*ignoreZPCheck=*/false, innerLog);

    const auto outQuantizeElemType = getQuantizedType(
            outLowConst.getContentAttr(), outHighConst.getContentAttr(), origOp.getLevels(), origOp.getLowFpType(),
            realElemType, false, origOp.getLoc(), origOp.getAutoBroadcast(), /*ignoreZPCheck=*/false, innerLog);

    innerLog.trace("Insert Quantize op '{0}' -> '{1}'", realElemType, inQuantizeElemType);
    auto quantizeOp = rewriter.create<IE::QuantizeOp>(origOp.getLoc(), origOp.getInput(), inQuantizeElemType);

    auto result = quantizeOp.getResult();
    if (inQuantizeElemType != outQuantizeElemType) {
        innerLog.trace("Insert QuantizeCast op '{0}' -> '{1}'", inQuantizeElemType, outQuantizeElemType);
        auto quantizeCastOp = rewriter.create<IE::QuantizeCastOp>(origOp.getLoc(), result, outQuantizeElemType);
        result = quantizeCastOp.getResult();
    }

    innerLog.trace("Insert Dequantize op '{0}' -> '{1}'", outQuantizeElemType, realElemType);
    rewriter.replaceOpWithNewOp<IE::DequantizeOp>(origOp, result, realElemType);

    return mlir::success();
}

//
// UseConstDequant
//

class UseConstDequant final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    UseConstDequant(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
        setDebugName("UseConstDequant");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::FailureOr<float> getCommonRatio(const Const::Content& content1, const Const::Content& content2) {
    const auto vals1 = content1.getValues<float>();
    const auto vals2 = content2.getValues<float>();

    if (vals1.size() != vals2.size()) {
        return mlir::failure();
    }

    if (std::equal(vals1.begin(), vals1.end(), vals2.begin(), isFloatEqual)) {
        return 1.0f;
    }

    SmallVector<float> ratios;
    ratios.reserve(vals1.size());

    std::transform(vals1.begin(), vals1.end(), vals2.begin(), std::back_inserter(ratios), std::divides<>{});

    // check that all ratios are equal
    if (std::adjacent_find(ratios.begin(), ratios.end(), [](float a, float b) {
            return !isFloatEqual(a, b);
        }) == ratios.end()) {
        return ratios[0];
    } else {
        // Input and output limits has per channel ratio
        return mlir::failure();
    }
}

bool hasMultiZeroPoint(IE::FakeQuantizeOp fqOp) {
    auto inLowConst = fqOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = fqOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = fqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();
    if (inLowConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        return false;
    }

    const auto inLowAttr = inLowConst.getContentAttr().fold();
    const auto outLowAttr = outLowConst.getContentAttr().fold();
    const auto outHighAttr = outHighConst.getContentAttr().fold();

    const auto outLowVals = outLowAttr.getValues<double>();
    const auto outHighVals = outHighAttr.getValues<double>();

    SmallVector<double> outLows(outLowVals);
    SmallVector<double> outHighs(outHighVals);

    const auto broadcast = fqOp.getAutoBroadcast();
    broadcastRange(outLows, outHighs, broadcast);

    mlir::Type storageType;
    int64_t qMin = 0;
    int64_t qMax = 0;
    const auto levels = fqOp.getLevels();
    const auto lowFpType = fqOp.getLowFpType();
    const auto isSigned = Const::hasNegativeValues(inLowAttr);
    if (levels.has_value()) {
        std::tie(qMin, qMax, storageType) = getStorageParams(inLowConst.getContext(), *levels, isSigned);
    } else if (lowFpType.has_value()) {
        std::tie(qMin, qMax, storageType) = getStorageParams(inLowConst.getContext(), *lowFpType);
    } else {
        VPUX_THROW("FakeQuantize op doesn't have levels and lowFpType");
    }

    SmallVector<int64_t> outZeroPoints(outLows.size());

    // unused variable
    double outScale = 0.0;
    loop_1d(LoopExecPolicy::Parallel, outLowConst.getContext(), outLows.size(), [&](size_t i) {
        std::tie(outScale, outZeroPoints[i]) = calcScaleAndZeroPoint(qMin, qMax, outLows[i], outHighs[i]);
    });

    return outLows.size() > 1 && !std::equal(outZeroPoints.begin() + 1, outZeroPoints.end(), outZeroPoints.begin());
}

mlir::LogicalResult UseConstDequant::matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got FakeQuantize Operation '{1}'", getDebugName(), origOp->getLoc());
    auto innerLog = _log.nest();

    auto inConst = origOp.getInput().getDefiningOp<Const::DeclareOp>();
    if (inConst == nullptr) {
        return matchFailed(innerLog, rewriter, origOp, "Got non constant input");
    }

    auto inLowConst = origOp.getInputLow().getDefiningOp<Const::DeclareOp>();
    auto inHighConst = origOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
    auto outLowConst = origOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
    auto outHighConst = origOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();

    if (inLowConst == nullptr || inHighConst == nullptr || outLowConst == nullptr || outHighConst == nullptr) {
        return matchFailed(innerLog, rewriter, origOp, "Got non constant parameters");
    }

    auto inConstAttr = inConst.getContentAttr();
    const auto inBaseVals = inConstAttr.getBaseContent();
    const auto inBaseElemType = inBaseVals.getShapedType().getElementType();
    const auto lowContent = inLowConst.getContentAttr().fold();
    auto isSigned = Const::hasNegativeValues(lowContent);

    // TODO: make this check more reliable
    if (!inBaseElemType.isa<mlir::IntegerType>()) {
        const auto inLowContent = inLowConst.getContent();
        const auto inHighContent = inHighConst.getContent();

        if (!inLowContent.isSplat() || !inHighContent.isSplat()) {
            innerLog.warning("Legacy model, original input values are not integer");

            // Workaround for Float weights, it lacks generality but is ok for old networks
            // Check if FQ can be removed for float weights
            const auto outLowContent = outLowConst.getContent();
            const auto outHighContent = outHighConst.getContent();

            const auto ratioLow = getCommonRatio(inLowContent, outLowContent);
            const auto ratioHigh = getCommonRatio(inHighContent, outHighContent);

            if (mlir::failed(ratioLow) || mlir::failed(ratioHigh)) {
                return matchFailed(innerLog, rewriter, origOp,
                                   "In and out limits differ and has per channel ratio, do not support");
            } else if (!isFloatEqual(ratioLow.value(), ratioHigh.value())) {
                return matchFailed(innerLog, rewriter, origOp, "Unsupported case, ratioHigh={0} != ratioLow={1}",
                                   ratioHigh, ratioLow);
            }

            if (ratioHigh.value() == 1.0f) {
                // FQ input and output ranges are equal, only remove FQ
                rewriter.replaceOpWithNewOp<Const::DeclareOp>(origOp, origOp.getType(), inConst.getContentAttr())
                        ->setLoc(inConst->getLoc());
            } else {
                // FQ input and output ranges are NOT equal, rescale weights
                innerLog.trace("Rescale weights");
                const auto newConstAttr = inConst.getContentAttr().rescale(ratioHigh.value());
                rewriter.replaceOpWithNewOp<Const::DeclareOp>(origOp, origOp.getType(), newConstAttr)
                        ->setLoc(inConst->getLoc());
            }

            return mlir::success();
        }

        // Check if FakeQuantize input range is equal to a low precision storage type and if not apply FakeQuantize
        // mathematical formula to the FakeQuantize input Const to quantize the data to the targeted low precision type
        // TODO: E#122705 Add Quantize transformation in Const dialect
        auto levels = origOp.getLevels();
        if (levels.has_value() && !isLowPrecisionTypeRange(getContext(), inLowContent.getValues<float>(),
                                                           inHighContent.getValues<float>(), *levels, isSigned)) {
            VPUX_THROW_WHEN(!inLowContent.isSplat() || !inHighContent.isSplat(), "Unsupported per axis case.");
            // Bring the constant values in low precision storage type range
            int64_t intQLow = 0;
            int64_t intQHigh = 0;
            std::tie(intQLow, intQHigh, std::ignore) = getStorageParams(getContext(), levels.value(), isSigned);
            auto qLow = checked_cast<float>(intQLow);
            auto qHigh = checked_cast<float>(intQHigh);

            auto inLow = inLowContent.getSplatValue<float>();
            auto inHigh = inHighContent.getSplatValue<float>();
            const auto inConstContent = inConst.getContent();
            const auto inVals = inConstContent.getValues<float>();
            SmallVector<float> quantizedVals(inVals.size());
            float fLevels = checked_cast<float>(levels.value());
            for (size_t i = 0; i < inVals.size(); ++i) {
                quantizedVals[i] = fakeQuantize(inVals[i], inLow, inHigh, qLow, qHigh, fLevels);
            }

            // Generate the Const::ContentAttr with the adjusted constant content
            const auto inConstStorageType = inConstContent.getType().dyn_cast<mlir::RankedTensorType>();
            const auto quantizedConstElementVal = wrapData(inConstStorageType, quantizedVals);
            auto quantizedConstContentAttr = Const::ContentAttr::get(quantizedConstElementVal);
            inConstAttr = quantizedConstContentAttr;
        }
    }

    innerLog.trace("Try to use constant dequantize");

    const auto realType = inConstAttr.getType().cast<vpux::NDTypeInterface>();
    const auto realElemType = realType.getElementType().cast<mlir::FloatType>();

    const auto multiZeroPoint = hasMultiZeroPoint(origOp);
    const auto qElemType =
            getQuantizedType(outLowConst.getContentAttr(), outHighConst.getContentAttr(), origOp.getLevels(),
                             origOp.getLowFpType(), realElemType, Const::hasNegativeValues(lowContent), origOp.getLoc(),
                             origOp.getAutoBroadcast(), multiZeroPoint, innerLog);

    if (qElemType == nullptr) {
        return mlir::failure();
    }

    innerLog.trace("Use quantized element type '{0}'", qElemType);

    const auto qType = realType.changeElemType(qElemType);

    auto newInConstAttr = inConstAttr.convertElemType(normalizeQuantStorageType(qElemType)).quantCast(qElemType);

    // Fuse dequantize to const directly since it could not convert to HW for multi Zero Point case
    if (multiZeroPoint) {
        newInConstAttr = newInConstAttr.dequantize();
        rewriter.replaceOpWithNewOp<Const::DeclareOp>(origOp, origOp.getType(), newInConstAttr)
                ->setLoc(inConst->getLoc());
        return mlir::success();
    }

    auto newInOp = rewriter.create<Const::DeclareOp>(inConst->getLoc(), qType, newInConstAttr);
    rewriter.replaceOpWithNewOp<IE::DequantizeOp>(origOp, newInOp.getOutput(), realElemType);
    return mlir::success();
}

//
// SplitFakeQuantPass
//

class SplitFakeQuantPass final : public IE::SplitFakeQuantBase<SplitFakeQuantPass> {
public:
    explicit SplitFakeQuantPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SplitFakeQuantPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);

    // per-channel quantization with different zero points is not supported on HW (E#65130)
    // if this is the case, the FQ op will be marked legal and will later be executed as SW op
    target.addDynamicallyLegalOp<IE::FakeQuantizeOp>([](IE::FakeQuantizeOp fqOp) {
        auto maybeConstOp = fqOp.getInput().getDefiningOp<Const::DeclareOp>();
        bool isConstInput = (maybeConstOp != nullptr);

        // #E-122320 support fuse range withoutZero to const.
        if (hasRangeWithoutZero(fqOp) && !isScalar(fqOp)) {
            return true;
        }

        auto inLowConst = fqOp.getInputLow().getDefiningOp<Const::DeclareOp>();
        auto inHighConst = fqOp.getInputHigh().getDefiningOp<Const::DeclareOp>();
        auto outLowConst = fqOp.getOutputLow().getDefiningOp<Const::DeclareOp>();
        auto outHighConst = fqOp.getOutputHigh().getDefiningOp<Const::DeclareOp>();
        const auto realType = fqOp.getInput().getType().cast<vpux::NDTypeInterface>();
        const auto realElemType = realType.getElementType().cast<mlir::FloatType>();

        // Although HW could not support multi Zero Point, but if the input is const, we can fuse the fq to const
        // So here ignore multi Zero Point check for const input
        const auto inQuantizeElemType = getQuantizedType(
                inLowConst.getContentAttr(), inHighConst.getContentAttr(), fqOp.getLevels(), fqOp.getLowFpType(),
                realElemType, false, fqOp.getLoc(), fqOp.getAutoBroadcast(), /*ignoreZPCheck=*/isConstInput);

        if (inQuantizeElemType == nullptr)
            return true;

        const auto outQuantizeElemType = getQuantizedType(
                outLowConst.getContentAttr(), outHighConst.getContentAttr(), fqOp.getLevels(), fqOp.getLowFpType(),
                realElemType, false, fqOp.getLoc(), fqOp.getAutoBroadcast(), /*ignoreZPCheck=*/isConstInput);

        return outQuantizeElemType == nullptr;
    });

    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::QuantizeOp>();
    target.addLegalOp<IE::QuantizeCastOp>();
    target.addLegalOp<IE::DequantizeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<UseQuantDequant>(&ctx, _log);
    patterns.add<UseConstDequant>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSplitFakeQuantPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSplitFakeQuantPass(Logger log) {
    return std::make_unique<SplitFakeQuantPass>(log);
}
