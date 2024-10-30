//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

class FoldTileOpWithSingleValueInput final : public mlir::OpRewritePattern<IE::TileOp> {
public:
    FoldTileOpWithSingleValueInput(mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpRewritePattern<IE::TileOp>(ctx), _log(log) {
        setDebugName("FoldTileOpWithSingleValueInput");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TileOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const Logger& _log;
};

mlir::LogicalResult FoldTileOpWithSingleValueInput::matchAndRewrite(IE::TileOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    auto ctx = getContext();

    auto origInputType = origOp.getInput().getType().cast<NDTypeInterface>();
    auto origInputShape = origInputType.getShape();
    if (origInputShape.totalSize() != 1) {
        return mlir::failure();
    }

    if (!origOp->hasOneUse()) {
        return mlir::failure();
    }

    const auto isFoldableViewOp = [](mlir::Operation* viewOp) {
        if (!mlir::isa<IE::ReshapeOp, IE::AffineReshapeOp, IE::ShapeCastOp>(viewOp)) {
            return false;
        }
        if (!viewOp->hasOneUse()) {
            return false;
        }
        return true;
    };

    auto outputValue = origOp.getOutput().cast<mlir::Value>();
    auto outputUserOp = *(outputValue.getUsers().begin());
    while (isFoldableViewOp(outputUserOp)) {
        outputValue = outputUserOp->getResult(0);
        outputUserOp = *(outputValue.getUsers().begin());
    }

    // More ops which support auto broadcast may also apply here!
    if (!mlir::isa_and_nonnull<IE::MultiplyOp, IE::AddOp>(outputUserOp)) {
        return mlir::failure();
    }

    // Can't fold TileOp if the layer has post operation
    if (auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(outputUserOp)) {
        if (layerWithPostOp.getPostOp().has_value()) {
            return mlir::failure();
        }
    }

    _log.trace("Folding TileOp at '{0}'", origOp.getLoc());

    auto newShape = SmallVector<int64_t>(outputValue.getType().cast<NDTypeInterface>().getRank(), 1);
    auto newReshapeOp = rewriter.createOrFold<IE::ReshapeOp>(origOp.getLoc(), origOp.getInput(), nullptr, false,
                                                             getIntArrayAttr(ctx, newShape));

    outputValue.replaceAllUsesWith(newReshapeOp);

    return mlir::success();
}

//
// OptimizeTileOpPass
//

class OptimizeTileOpPass final : public IE::OptimizeTileOpBase<OptimizeTileOpPass> {
public:
    explicit OptimizeTileOpPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void OptimizeTileOpPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FoldTileOpWithSingleValueInput>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeTileOpPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeTileOpPass(Logger log) {
    return std::make_unique<OptimizeTileOpPass>(log);
}
