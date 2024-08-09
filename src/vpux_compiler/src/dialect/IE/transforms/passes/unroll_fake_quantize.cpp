//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

class UnrollFakeQuantize final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    UnrollFakeQuantize(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
        setDebugName("UnrollFakeQuantize");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::Value getValue(const mlir::ValueRange values, const int64_t idx) const;
    SmallVector<mlir::Value> splitValue(const mlir::Value val, const int64_t axis,
                                        mlir::PatternRewriter& rewriter) const;
    SmallVector<mlir::Value> splitInputs(IE::FakeQuantizeOp fqOp, const int64_t axis,
                                         mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

// Dispatch the case when the shapes of quantization parameters don't match.
// For example fqLow = 1x1x1, fqHigh = 16x1x64.
// In that case fqHigh is split into an array of 16 tensors.
// fqLow array will contain only one value.
mlir::Value UnrollFakeQuantize::getValue(const mlir::ValueRange values, const int64_t idx) const {
    if (values.size() == 1) {
        return values[0];
    }
    VPUX_THROW_UNLESS(idx < checked_cast<int64_t>(values.size()), "Out of bounds access: index: {0}, values: {1}", idx,
                      values.size());
    return values[idx];
}

// Split a value by specified axis.
// 1x3x1x16 with axis 1 will be split in three 1x1x1x16 values
// 1x1x2x16 with axis 2 will be split in two 1x1x1x16 values
// 1x3x1x16 with axis 2 won't be split and will return a vector with only one 1x3x1x16 element
SmallVector<mlir::Value> UnrollFakeQuantize::splitValue(const mlir::Value val, const int64_t axis,
                                                        mlir::PatternRewriter& rewriter) const {
    const auto valShape = getShape(val);
    VPUX_THROW_UNLESS(axis < checked_cast<int64_t>(valShape.size()), "Cannot split shape {0} by axis {1}", valShape,
                      axis);
    const auto groups = valShape[Dim(axis)];
    if (groups == 1) {
        return SmallVector<mlir::Value>{val};
    }
    auto staticSizes = to_small_vector(valShape);
    staticSizes[axis] = 1;
    const auto staticSizesAttr = getIntArrayAttr(rewriter.getContext(), staticSizes);
    SmallVector<mlir::Value> inputChunks;
    for (const auto& idx : irange(groups)) {
        const auto loc = appendLoc(val.getLoc(), "_slice_{0}", idx);
        SmallVector<int64_t> offsets(valShape.size(), 0);
        offsets[axis] = idx;
        const auto offsetsAttr = getIntArrayAttr(rewriter.getContext(), offsets);
        auto slice = rewriter.create<IE::SliceOp>(loc, val, offsetsAttr, staticSizesAttr);

        inputChunks.push_back(slice.getResult());
    }
    return inputChunks;
}

// Split the inputs (data, input_high, input_low, output_high, output_low) by specified axis.
// The method returns a vector of fake quantize operations.
// IE.FakeQuantize with data = 1x2x8x16, in_low = in_high = 1x1x1x1, out_low = out_high = 1x2x1x16
// will be split by channels into 2 FQ operations:
// IE.FakeQuantize with data = 1x1x8x16, in_low = in_high = 1x1x1x1, out_low = out_high = 1x1x1x16
SmallVector<mlir::Value> UnrollFakeQuantize::splitInputs(IE::FakeQuantizeOp fqOp, const int64_t axis,
                                                         mlir::PatternRewriter& rewriter) const {
    const auto data = splitValue(fqOp.getInput(), axis, rewriter);
    const auto inLow = splitValue(fqOp.getInputLow(), axis, rewriter);
    const auto inHigh = splitValue(fqOp.getInputHigh(), axis, rewriter);
    const auto outLow = splitValue(fqOp.getOutputLow(), axis, rewriter);
    const auto outHigh = splitValue(fqOp.getOutputHigh(), axis, rewriter);
    SmallVector<mlir::Value> fqResults;
    const auto groups = data.size();
    for (const auto& idx : irange(groups)) {
        const auto loc = appendLoc(fqOp.getLoc(), "_slice_{0}", idx);
        auto reducedFq = rewriter.create<IE::FakeQuantizeOp>(
                loc, data[idx], getValue(inLow, idx), getValue(inHigh, idx), getValue(outLow, idx),
                getValue(outHigh, idx), fqOp.getLevelsAttr(), fqOp.getLowFpTypeAttr(), fqOp.getAutoBroadcast());
        fqResults.push_back(reducedFq.getResult());
    }
    return fqResults;
}

mlir::LogicalResult UnrollFakeQuantize::matchAndRewrite(IE::FakeQuantizeOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    if (mlir::isa_and_nonnull<IE::GroupConvolutionOp>(*origOp.getResult().user_begin())) {
        return mlir::failure();
    }

    const auto axes = IE::findAxes(origOp);
    // Cases when there's only one axis or there are no axes at all are fine. Nothing to do.
    if (axes.size() <= 1) {
        return mlir::failure();
    }
    // Unroll the outermost axis. std::set must contain a sorted set of elements.
    const auto axis = *axes.begin();
    const auto fqChunks = splitInputs(origOp, axis, rewriter);
    auto concatOp = rewriter.create<IE::ConcatOp>(origOp->getLoc(), fqChunks, axis);
    rewriter.replaceOp(origOp, concatOp.getResult());
    return mlir::success();
}

class UnrollFakeQuantizePass final : public IE::UnrollFakeQuantizeBase<UnrollFakeQuantizePass> {
public:
    explicit UnrollFakeQuantizePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollFakeQuantizePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<UnrollFakeQuantize>(&ctx, _log);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createUnrollFakeQuantizePass(Logger log) {
    return std::make_unique<UnrollFakeQuantizePass>(log);
}
