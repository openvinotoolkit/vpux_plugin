//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/convert_to_mixed_precision.hpp"
#include "vpux/compiler/VPU30XX/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
using namespace vpux;

namespace {

bool isMixPrecisionSupported(mlir::Operation* origOp, const bool isPReLUSupported, Logger log) {
    if (!mlir::isa<IE::ConvolutionOp, IE::GroupConvolutionOp, IE::AddOp>(origOp)) {
        return false;
    }
    if (isPReLUSupported) {
        return false;
    }
    // Check that the kernel size are not exceding the NCE HW limits
    if (VPU::NCEInvariant::verifyKernel(origOp, log).failed()) {
        return false;
    }

    // If the Add operands have different shapes the operation will be mapped on SHAVE, which does not support mixed
    // precision operations
    if (mlir::isa<IE::AddOp>(origOp)) {
        auto addOp = mlir::dyn_cast<IE::AddOp>(origOp);
        const auto shape1 = getShape(addOp.getInput1());
        const auto shape2 = getShape(addOp.getInput2());
        if (shape1 != shape2)
            return false;
    }

    // Mixed precision for average pooling is not supported for VPUX30XX target
    if (mlir::isa<IE::AvgPoolOp>(origOp)) {
        return false;
    }

    // NOTE: HW limitation, in mixed mode the grids of the MPEs are conflicting between
    // each other, which leads to 1x1 workloads.
    auto outputShape = getShape(origOp->getResult(0));
    return outputShape[Dims4D::Act::H] == 1 && outputShape[Dims4D::Act::W] == 1;
}

//
// ConvertToMixedPrecisionPass
//

class ConvertToMixedPrecisionPass final :
        public IE::arch30xx::ConvertToMixedPrecisionBase<ConvertToMixedPrecisionPass> {
public:
    explicit ConvertToMixedPrecisionPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ConvertToMixedPrecisionPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    // E#67754 - MaxPool is omitted intentionally because it generates accuracy issues.
    patterns.add<vpux::IE::FloatOutConvRewriter>(&ctx, isMixPrecisionSupported, _log);
    patterns.add<vpux::IE::FloatOutGroupConvRewriter>(&ctx, isMixPrecisionSupported, _log);
    patterns.add<vpux::IE::FloatOutAddRewriter>(&ctx, isMixPrecisionSupported,
                                                false /*VPU30XX does not support different scales*/, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertToMixedPrecision
//

std::unique_ptr<mlir::Pass> vpux::IE::arch30xx::createConvertToMixedPrecision(Logger log) {
    return std::make_unique<ConvertToMixedPrecisionPass>(log);
}
