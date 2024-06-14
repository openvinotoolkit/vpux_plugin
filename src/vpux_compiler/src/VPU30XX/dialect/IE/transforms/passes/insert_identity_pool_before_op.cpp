//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/insert_identity_pool_before_op.hpp"
#include "vpux/compiler/VPU30XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace vpux;

namespace {

mlir::Operation* insertMaxPool(mlir::Operation* concreteOp, mlir::PatternRewriter& rewriter, Logger log) {
    if (!IE::isEligiblePostOp(concreteOp, log)) {
        return nullptr;
    }

    const auto outputType = concreteOp->getOperand(0).getType();
    return IE::createIdentityMaxPool(concreteOp->getOperand(0), outputType, rewriter);
}

//
// InsertIdentityPoolBeforeOpPass
//

class InsertIdentityPoolBeforeOpPass final :
        public IE::arch30xx::InsertIdentityPoolBeforeOpBase<InsertIdentityPoolBeforeOpPass> {
public:
    explicit InsertIdentityPoolBeforeOpPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void InsertIdentityPoolBeforeOpPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<IE::InsertIdPoolRewriter<IE::LeakyReluOp>>(&ctx, insertMaxPool, _log);
    patterns.add<IE::InsertIdPoolRewriter<IE::ClampOp>>(&ctx, insertMaxPool, _log);
    patterns.add<IE::InsertIdPoolRewriter<IE::ReLUOp>>(&ctx, insertMaxPool, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::arch30xx::createInsertIdentityPoolBeforeOpPass(Logger log) {
    return std::make_unique<InsertIdentityPoolBeforeOpPass>(log);
}
