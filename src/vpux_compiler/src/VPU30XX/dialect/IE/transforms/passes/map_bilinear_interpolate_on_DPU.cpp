//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU30XX/dialect/IE/transforms/passes/map_bilinear_interpolate_on_DPU.hpp"
#include "vpux/compiler/VPU30XX/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

// Creates identify pooling operations which to ensure that all the inputs of the Concat operations that
// compose the scaled results on each axis has only NCE inputs
// This ensures that the compiler will map more optimal the generated operations on the available HW resources
mlir::Value IE::arch30xx::MapBilinearInterpolateOnDPURewriter::createIdentityPooling(mlir::PatternRewriter& rewriter,
                                                                                     mlir::Location loc,
                                                                                     mlir::Value input) const {
    const SmallVector<int64_t> poolStrides = {1, 1};
    const SmallVector<int64_t> poolKernels = {1, 1};
    const SmallVector<int64_t> pads = {0, 0};
    const auto padsAttr = getIntArrayAttr(rewriter, pads);

    auto maxPoolOp = rewriter.create<IE::MaxPoolOp>(
            loc, input, getIntArrayAttr(rewriter, poolKernels), getIntArrayAttr(rewriter, poolStrides), padsAttr,
            padsAttr, vpux::IE::RoundingTypeAttr::get(rewriter.getContext(), vpux::IE::RoundingType::FLOOR), nullptr,
            nullptr);

    return maxPoolOp.getOutput();
}

namespace {

//
// MapBilinearInterpolateOnDPUPass
//

class MapBilinearInterpolateOnDPUPass final :
        public IE::arch30xx::MapBilinearInterpolateOnDPUPassBase<MapBilinearInterpolateOnDPUPass> {
public:
    explicit MapBilinearInterpolateOnDPUPass(const bool interpolateAsSEOp, Logger log)
            : _interpolateAsSEOp(interpolateAsSEOp) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

public:
    class MapBilinearInterpolateOnDPURewriter;

private:
    void safeRunOnFunc() final;

private:
    bool _interpolateAsSEOp;
};

mlir::LogicalResult MapBilinearInterpolateOnDPUPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (interpolateAsSEOp.hasValue()) {
        _interpolateAsSEOp = interpolateAsSEOp.getValue();
    }

    return mlir::success();
}

void MapBilinearInterpolateOnDPUPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::InterpolateOp>([&](IE::InterpolateOp op) {
        return isLegalInterpolateOp(op, _interpolateAsSEOp, logCb);
    });

    target.addLegalOp<IE::ExpandOp>();
    target.addLegalOp<IE::MaxPoolOp>();
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::GroupConvolutionOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<IE::arch30xx::MapBilinearInterpolateOnDPURewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMapBilinearInterpolateOnDPUPass
//

std::unique_ptr<mlir::Pass> vpux::IE::arch30xx::createMapBilinearInterpolateOnDPUPass(const bool interpolateAsSEOp,
                                                                                      Logger log) {
    return std::make_unique<MapBilinearInterpolateOnDPUPass>(interpolateAsSEOp, log);
}
