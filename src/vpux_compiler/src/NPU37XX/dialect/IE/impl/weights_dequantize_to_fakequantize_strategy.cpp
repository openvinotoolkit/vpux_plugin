//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/weights_dequantize_to_fakequantize_strategy.hpp"
#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"

using namespace vpux;

namespace {

mlir::LogicalResult commonMatchAndRewrite(IE::WeightsDequantizeStructureInfo& wdInfo, mlir::PatternRewriter& rewriter) {
    if (!wdInfo.isSuccessfulMatch()) {
        wdInfo.log.trace("Failed to match WeightsDequantize structure");
        return mlir::failure();
    }

    // The only supported weights data type are I8, U8, I4 and U4
    if (!wdInfo.has8BitIntegerInput() && !wdInfo.has4BitIntegerInput()) {
        wdInfo.log.trace("Input data type {0} is not supported.", wdInfo.getInputElemStorageType());
        return mlir::failure();
    }

    // Expensive cast to high precision, done only after all validations pass
    if (wdInfo.hasConstInput() && mlir::failed(wdInfo.ensureHighPrecisionStorage())) {
        wdInfo.log.trace("Failed to cast low precision weights to high precision");
        return mlir::failure();
    }

    // Compute input low, input high constants of FakeQuantize using the value interval of the weights type
    const auto levels = wdInfo.getQuantizationLevels();
    const auto levelsAttr = getIntAttr(wdInfo.getContext(), levels);
    const float inLow = (wdInfo.hasSignedInput() ? -(levels / 2) : 0);
    const float inHigh = (levels + inLow - 1);

    const auto inInterval = wdInfo.getInputQuantizationInterval(inLow, inHigh);
    auto inLowConst =
            rewriter.create<Const::DeclareOp>(wdInfo.getLocation(), inInterval.first.getType(), inInterval.first);
    auto inHighConst =
            rewriter.create<Const::DeclareOp>(wdInfo.getLocation(), inInterval.second.getType(), inInterval.second);

    // Compute output low and output high constants of FakeQuantize by applying a reverse scale-shift to the inputs
    const auto outInterval = wdInfo.getOutputQuantizationInterval(inInterval);
    auto outLowConst =
            rewriter.create<Const::DeclareOp>(wdInfo.getLocation(), outInterval.first.getType(), outInterval.first);
    auto outHighConst =
            rewriter.create<Const::DeclareOp>(wdInfo.getLocation(), outInterval.second.getType(), outInterval.second);

    const auto broadCastAttr = IE::AutoBroadcastTypeAttr::get(wdInfo.getContext(), IE::AutoBroadcastType::NUMPY);

    // Create the FakeQuantize to replace the WD pattern (since we're working with intergral input, only levelsAttr is
    // given)
    if (wdInfo.hasConstInput()) {
        // redeclare the input after a potential high precision storage cast (this also removes any
        // ConvertElemType transforms)
        Const::DeclareOp fakeQuantizeInput = rewriter.create<Const::DeclareOp>(
                wdInfo.getLocation(), wdInfo.getInputShapedType(), wdInfo.getInputContentAttr());

        rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(wdInfo.getLastOp(), fakeQuantizeInput, inLowConst, inHighConst,
                                                        outLowConst, outHighConst, levelsAttr, /*lowFpType=*/nullptr,
                                                        broadCastAttr);
    } else {
        // without this the FQ insertion occurs before ConvertOp (=> error)
        rewriter.setInsertionPoint(wdInfo.getLastOp());
        rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(wdInfo.getLastOp(), wdInfo.getInputValue(), inLowConst,
                                                        inHighConst, outLowConst, outHighConst, levelsAttr,
                                                        /*lowFpType=*/nullptr, broadCastAttr);
    }

    return mlir::success();
}

class WeightsDequantizeToFakeQuantizeConstRewriter final : public mlir::OpRewritePattern<Const::DeclareOp> {
public:
    WeightsDequantizeToFakeQuantizeConstRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<Const::DeclareOp>(ctx), _log(log) {
        setDebugName("WeightsDequantizeToFakeQuantizeRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(Const::DeclareOp origOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("Got {0} at `{1}`.", origOp->getName(), origOp->getLoc());

        IE::WeightsDequantizeStructureInfo wdInfo(origOp, _log.nest());
        return commonMatchAndRewrite(wdInfo, rewriter);
    }

private:
    Logger _log;
};

class WeightsDequantizeToFakeQuantizeBlockRewriter final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    WeightsDequantizeToFakeQuantizeBlockRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
        setDebugName("WeightsDequantizeToFakeQuantizeRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("Got {0} at `{1}`.", origOp->getName(), origOp->getLoc());

        IE::WeightsDequantizeStructureInfo wdInfo(origOp, _log.nest());
        return commonMatchAndRewrite(wdInfo, rewriter);
    }

private:
    Logger _log;
};

}  // namespace

//
// WeightsDequantizeToFakeQuantizeStrategy
//

IE::arch37xx::WeightsDequantizeToFakeQuantizeStrategy::WeightsDequantizeToFakeQuantizeStrategy(
        bool enableWDBlockArgumentInput) noexcept
        : _enableWDBlockArgumentInput(enableWDBlockArgumentInput) {
}

void IE::arch37xx::WeightsDequantizeToFakeQuantizeStrategy::addPatterns(mlir::RewritePatternSet& patterns,
                                                                        Logger& log) const {
    auto ctx = patterns.getContext();

    IE::ConvertOp::getCanonicalizationPatterns(patterns, ctx);

    patterns.add<WeightsDequantizeToFakeQuantizeConstRewriter>(ctx, log);
    if (_enableWDBlockArgumentInput) {
        patterns.add<WeightsDequantizeToFakeQuantizeBlockRewriter>(ctx, log);
    }
}
