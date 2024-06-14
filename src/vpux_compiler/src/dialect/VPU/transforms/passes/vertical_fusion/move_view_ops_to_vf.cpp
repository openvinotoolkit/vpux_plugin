//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

#include "vpux/compiler/utils/types.hpp"

#include <mlir/Pass/PassManager.h>

using namespace vpux;
using namespace VPU;

namespace {

//
// ViewOpsRewriter
//

class ViewOpsRewriter final : public mlir::OpRewritePattern<VPU::VerticalFusionOp> {
public:
    ViewOpsRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::VerticalFusionOp>(ctx), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPU::VerticalFusionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ViewOpsRewriter::matchAndRewrite(VPU::VerticalFusionOp vfOp,
                                                     mlir::PatternRewriter& rewriter) const {
    for (auto vfOperand : vfOp->getOperands() | indexed) {
        auto parentOp = vfOperand.value().getDefiningOp<VPU::TilingViewLikeOpInterface>();

        if (!parentOp || !VPU::isPureViewOp(parentOp)) {
            continue;
        }

        if (llvm::all_of(parentOp->getOperands(), [](auto value) {
                return mlir::isa_and_nonnull<mlir::BlockArgument>(value) ||
                       mlir::isa_and_nonnull<Const::DeclareOp>(value.getDefiningOp());
            })) {
            continue;
        }

        auto newVFOp = fuseOpsInBlock(rewriter, vfOp, parentOp);
        rewriter.replaceOp(vfOp, newVFOp.getResult(0));
        return mlir::success();
    }
    return mlir::failure();
}

//
// MoveViewOpsToVFPass
//

class MoveViewOpsToVFPass final : public MoveViewOpsToVFBase<MoveViewOpsToVFPass> {
public:
    explicit MoveViewOpsToVFPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnModule
//

void MoveViewOpsToVFPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ViewOpsRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                                        getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMoveViewOpsToVerticalFusionPass
//

std::unique_ptr<mlir::Pass> VPU::createMoveViewOpsToVerticalFusionPass(Logger log) {
    return std::make_unique<MoveViewOpsToVFPass>(log);
}
