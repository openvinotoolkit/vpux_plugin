//
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/factories/weights_dequantize_to_fakequantize_strategy_getter.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <vector>

namespace vpux {

class WeightsDequantizeToFakeQuantizePass final :
        public IE::WeightsDequantizeToFakeQuantizeBase<WeightsDequantizeToFakeQuantizePass> {
public:
    explicit WeightsDequantizeToFakeQuantizePass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void WeightsDequantizeToFakeQuantizePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);

    // register platform specific rewriters using the platform specific strategy
    auto strategy = vpux::IE::createWeightsDequantizeToFakeQuantizeStrategyGetter(func);
    strategy->addPatterns(patterns, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace vpux

//
// createWeightsDequantizeToFakeQuantizePass
//
std::unique_ptr<mlir::Pass> vpux::IE::createWeightsDequantizeToFakeQuantizePass(Logger log) {
    return std::make_unique<WeightsDequantizeToFakeQuantizePass>(log);
}
