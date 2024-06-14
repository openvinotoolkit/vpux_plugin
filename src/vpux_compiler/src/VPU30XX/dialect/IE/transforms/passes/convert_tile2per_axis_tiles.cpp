//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU30XX/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ConvertTile2PerAxisTile
//

class ConvertTile2PerAxisTilePass final :
        public IE::arch30xx::ConvertTile2PerAxisTileBase<ConvertTile2PerAxisTilePass> {
public:
    explicit ConvertTile2PerAxisTilePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class TileOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// GenericOpConverter
//

class ConvertTile2PerAxisTilePass::TileOpConverter final : public mlir::OpRewritePattern<IE::TileOp> {
public:
    TileOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::TileOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TileOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertTile2PerAxisTilePass::TileOpConverter::matchAndRewrite(
        IE::TileOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got Tile Operation '{0}'", origOp->getLoc());

    if (!origOp.getRepeatsValues().has_value()) {
        return errorAt(origOp->getLoc(), "Got non repeats_values parameters");
    }
    auto repeats = parseIntArrayAttr<int64_t>(origOp.getRepeatsValues().value());

    const auto gapSize = static_cast<int>(getShape(origOp.getInput()).size()) - static_cast<int>(repeats.size());

    if (gapSize < 0) {
        _log.error("Op: {0}. Rank of input tensor less then input repeats array size. This case should be handled by "
                   "canonicalizer of TileOp",
                   origOp->getLoc());
        return mlir::failure();
    }

    mlir::Value lastResult = origOp.getInput();
    for (size_t i = 0; i < repeats.size(); ++i) {
        if (repeats[i] > 1) {
            lastResult = rewriter.create<IE::PerAxisTileOp>(origOp->getLoc(), lastResult, i + gapSize, repeats[i])
                                 .getResult();
        }
    }
    if (lastResult == origOp.getInput()) {
        return mlir::failure();
    }
    rewriter.replaceOp(origOp, lastResult);
    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertTile2PerAxisTilePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::TileOp>();
    target.addLegalOp<IE::PerAxisTileOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<TileOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertTile2PerAxisTilePass
//

std::unique_ptr<mlir::Pass> vpux::IE::arch30xx::createConvertTile2PerAxisTilePass(Logger log) {
    return std::make_unique<ConvertTile2PerAxisTilePass>(log);
}
