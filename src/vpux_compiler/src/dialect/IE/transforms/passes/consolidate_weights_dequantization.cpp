//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <cstdint>

namespace vpux {

class DynamicDequantRewriter final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    DynamicDequantRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
        setDebugName("DynamicDequantRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertOp origOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("Got {0} at `{1}`.", origOp->getName(), origOp->getLoc());
        auto maybeWdInfo = IE::WeightsDequantizeStructureInfo::create(origOp, _log.nest());
        if (mlir::failed(maybeWdInfo)) {
            _log.trace("Failed to match WeightsDequantize structure");
            return mlir::failure();
        }
        auto wdInfo = maybeWdInfo.value();

        const auto loc = wdInfo.getLastOp()->getLoc();
        // The only supported weights data type for dynamic quantize is I4 and U4
        const auto inputElemType = wdInfo.getTrueElemTypeOfWeights();
        if (!inputElemType.isInteger(4)) {
            _log.trace("Input data type {0} is not supported.", inputElemType);
            return mlir::failure();
        }

        mlir::Value dynamicDequantInput = origOp.getInput();
        const auto oldOutput = wdInfo.getLastOp();
        rewriter.setInsertionPointAfter(origOp);

        const auto ctx = origOp->getContext();
        auto [int4Min, int4Max, int4Type] = getStorageParams(ctx, wdInfo.getQuantizationLevels(), true);

        int64_t shiftValue = 0;
        const auto shift = wdInfo.getShift();
        if (shift != nullptr) {
            if (!shift.isSplat()) {
                _log.trace("ZP is not scalar.");
                return mlir::failure();
            }
            shiftValue = shift.fold().getSplatValue<int64_t>();
        }

        auto filterElemType =
                mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::Signed, int4Type,
                                                       mlir::Float16Type::get(ctx), 1, shiftValue, int4Min, int4Max);
        auto inputValue = rewriter.create<IE::QuantizeCastOp>(loc, dynamicDequantInput, filterElemType).getOutput();
        if (auto transposeOp = mlir::dyn_cast_or_null<IE::TransposeOp>(wdInfo.getInput().getDefiningOp())) {
            inputValue = rewriter.create<IE::TransposeOp>(loc, inputValue, nullptr, transposeOp.getOrderValueAttr())
                                 .getOutput();
        }
        auto dynamicDequantizeOp =
                rewriter.create<IE::DynamicDequantizeOp>(appendLoc(loc, "artificial_dyn_dequant"), inputValue,
                                                         wdInfo.getDynamicScale(), nullptr, origOp.getDstElemType());
        oldOutput->replaceAllUsesWith(dynamicDequantizeOp);
        wdInfo.cleanUpCurrentWdChain(rewriter);
        return mlir::success();
    }

private:
    Logger _log;
};

class ConsolidateWeightsDequantizationPass final :
        public IE::ConsolidateWeightsDequantizationBase<ConsolidateWeightsDequantizationPass> {
public:
    explicit ConsolidateWeightsDequantizationPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConsolidateWeightsDequantizationPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);

    patterns.add<DynamicDequantRewriter>(&ctx, _log);
    auto config = getDefaultGreedyRewriteConfig();
    config.maxIterations = mlir::GreedyRewriteConfig::kNoLimit;
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
        signalPassFailure();
    }
}

}  // namespace vpux

//
// createConsolidateWeightsDequantizationPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConsolidateWeightsDequantizationPass(Logger log) {
    return std::make_unique<ConsolidateWeightsDequantizationPass>(log);
}
