//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/se_padding_utils.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ConvertReflectPadToSliceAndConcatPass
//

class ConvertReflectPadToSliceAndConcatPass final :
        public IE::ConvertReflectPadToSliceAndConcatBase<ConvertReflectPadToSliceAndConcatPass> {
public:
    explicit ConvertReflectPadToSliceAndConcatPass(const bool enableSEPPad, Logger log): _enableSEPPad(enableSEPPad) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

public:
    class ReflectPadConverter;

private:
    void safeRunOnFunc() final;
    bool _enableSEPPad;
};

mlir::LogicalResult ConvertReflectPadToSliceAndConcatPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (enableSEPPad.hasValue()) {
        _enableSEPPad = enableSEPPad.getValue();
    }

    return mlir::success();
}

//
// ReflectPadConverter
//

class ConvertReflectPadToSliceAndConcatPass::ReflectPadConverter final : public mlir::OpRewritePattern<IE::PadOp> {
public:
    ReflectPadConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::PadOp>(ctx), _log(log) {
        setDebugName("ReflectPadConverter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::PadOp PadOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::Value convertPerAxisPad(mlir::Value input, const int64_t padBegin, const int64_t padEnd, const Dim padAxis,
                              mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::MLIRContext* ctx) {
    if (padEnd == 0 && padBegin == 0) {
        return input;
    }

    auto inputShape = getShape(input);
    auto sliceData = [&](int64_t offset, StringRef locSuffix) {
        auto offsets = SmallVector<int64_t>(inputShape.size(), 0);
        auto sizes = SmallVector<int64_t>(inputShape.begin(), inputShape.end());
        offsets[padAxis.ind()] = offset;
        sizes[padAxis.ind()] = 1;
        return rewriter
                .create<IE::SliceOp>(appendLoc(loc, "slice_{0}_{1}", offset, locSuffix), input,
                                     getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, sizes))
                .getResult();
    };

    SmallVector<mlir::Value> subSlices;
    for (auto idx = padBegin; idx > 0; --idx) {
        const auto offset = idx;
        subSlices.push_back(sliceData(offset, "forward"));
    }

    subSlices.push_back(input);

    for (auto idx = 1; idx <= padEnd; ++idx) {
        const auto offset = inputShape[padAxis] - 1 - idx;
        subSlices.push_back(sliceData(offset, "backward"));
    }

    return rewriter.create<IE::ConcatOp>(loc, subSlices, padAxis).getOutput();
}

mlir::LogicalResult ConvertReflectPadToSliceAndConcatPass::ReflectPadConverter::matchAndRewrite(
        IE::PadOp padOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Found '{1}' at '{2}'", getDebugName(), padOp->getName(), padOp->getLoc());

    VPUX_THROW_UNLESS(padOp.getPadsBeginAttr().has_value() && padOp.getPadsEndAttr().has_value(),
                      "Cannot get pad begin and pad end value");
    const auto padsBegin = parseIntArrayAttr<int64_t>(padOp.getPadsBeginAttr().value());
    const auto padsEnd = parseIntArrayAttr<int64_t>(padOp.getPadsEndAttr().value());

    mlir::Value input = padOp.getInput();
    auto inputRank = input.getType().cast<mlir::RankedTensorType>().getRank();
    for (auto axis = inputRank - 1; axis >= 0; --axis) {
        auto newLoc = appendLoc(padOp.getLoc(), "pad_axis_{0}", Dim(axis));
        input = convertPerAxisPad(input, padsBegin[axis], padsEnd[axis], Dim(axis), rewriter, newLoc,
                                  padOp.getContext());
    }

    rewriter.replaceOp(padOp, input);

    _log.trace("Accomplished conversion of reflect pad to slice and concat");

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertReflectPadToSliceAndConcatPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::ConversionTarget target(ctx);

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    target.addDynamicallyLegalOp<IE::PadOp>([&](IE::PadOp op) -> bool {
        if (_enableSEPPad &&
            VPU::isSupportedSEPPadOp(op, logCb, /*checkLayout=*/false, /*checkChannelAlignment=*/false)) {
            _log.nest().trace("Pad Operation can be executed using SEP");
            return true;
        }

        return op.getMode() != IE::PadMode::REFLECT;
    });
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ReflectPadConverter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertReflectPadToSliceAndConcatPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertReflectPadToSliceAndConcatPass(const bool enableSEPPad, Logger log) {
    return std::make_unique<ConvertReflectPadToSliceAndConcatPass>(enableSEPPad, log);
}
