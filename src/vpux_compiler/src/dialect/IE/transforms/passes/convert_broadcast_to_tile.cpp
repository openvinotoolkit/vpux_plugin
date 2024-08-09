//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// ConvertBroadcastToTile
//

class ConvertBroadcastToTile final : public mlir::OpRewritePattern<IE::BroadcastOp> {
public:
    ConvertBroadcastToTile(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::BroadcastOp>(ctx), _log(log) {
        setDebugName("ConvertBroadcastToTile");
    }

    mlir::LogicalResult matchAndRewrite(IE::BroadcastOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertBroadcastToTile::matchAndRewrite(IE::BroadcastOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    const auto inputShape = to_small_vector(getShape(origOp.getInput()));
    const auto outputShape = to_small_vector(getShape(origOp.getOutput()));
    const auto broadcastType = origOp.getMode().value_or(IE::BroadcastType::NUMPY);
    SmallVector<int64_t> broadcastAxes;

    VPUX_THROW_UNLESS(inputShape.size() <= outputShape.size(), "Broadcast input rank {0} exceeds output rank {1}",
                      inputShape.size(), outputShape.size());

    // Finds the axes over which the broadcasting rules apply. For example:
    // NUMPY and BIDIRECTIONAL: input 16x1x1, output 1x16x50x50 will return the axes [0, 2, 3]
    // EXPLICIT:                input 16, output 1x16x50x50, axesMapping 1 will return the axes [0, 2, 3]
    if (broadcastType == IE::BroadcastType::BIDIRECTIONAL || broadcastType == IE::BroadcastType::NUMPY) {
        broadcastAxes = vpux::IE::getBroadcastAxesNumpyBidirectional(inputShape, outputShape);
    } else if (broadcastType == IE::BroadcastType::EXPLICIT) {
        auto axesMapping = IE::constInputToData(origOp.getLoc(), origOp.getAxesMapping()).value();
        broadcastAxes = vpux::IE::getBroadcastAxesExplicit(axesMapping, outputShape);
    }

    auto adjustedInputShape = inputShape;
    for (const auto& axis : broadcastAxes) {
        if (adjustedInputShape.size() < outputShape.size()) {
            adjustedInputShape.insert(adjustedInputShape.begin() + axis, 1);
        }
    }

    SmallVector<int32_t> repeats(outputShape.size());
    for (size_t i = 0; i < repeats.size(); ++i) {
        repeats[i] = static_cast<int32_t>(outputShape[i] / adjustedInputShape[i]);
    }

    const auto dataType =
            mlir::RankedTensorType::get({static_cast<int64_t>(repeats.size())}, getSInt32Type(getContext()));
    const auto repeatsConstOp = Const::createConst(rewriter, origOp->getLoc(), dataType, ArrayRef(repeats));

    const auto adjustedInputShapeAttr = getIntArrayAttr(getContext(), adjustedInputShape);
    auto reshapeInputOp =
            rewriter.create<IE::ReshapeOp>(origOp->getLoc(), origOp.getInput(), nullptr, false, adjustedInputShapeAttr);

    rewriter.replaceOpWithNewOp<IE::TileOp>(origOp, origOp.getType(), reshapeInputOp.getOutput(), repeatsConstOp,
                                            nullptr);

    return mlir::success();
}

//
// ConvertBroadcastToTilePass
//

class ConvertBroadcastToTilePass final : public IE::ConvertBroadcastToTileBase<ConvertBroadcastToTilePass> {
public:
    explicit ConvertBroadcastToTilePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertBroadcastToTilePass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::BroadcastOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::TileOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvertBroadcastToTile>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertBroadcastToTilePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertBroadcastToTilePass(Logger log) {
    return std::make_unique<ConvertBroadcastToTilePass>(log);
}
