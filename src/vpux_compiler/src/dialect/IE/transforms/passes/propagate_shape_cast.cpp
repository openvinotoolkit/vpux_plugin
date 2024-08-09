//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

//
// MoveThroughActivationOp
//

template <class ConcreteOp>
class MoveThroughActivationOp final : public mlir::OpRewritePattern<IE::ShapeCastOp> {
public:
    MoveThroughActivationOp(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ShapeCastOp>(ctx), _log(log) {
        this->setDebugName("MoveThroughActivationOp");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ShapeCastOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult MoveThroughActivationOp<ConcreteOp>::matchAndRewrite(IE::ShapeCastOp origOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    auto concreteOp = origOp.getOperand().getDefiningOp<ConcreteOp>();

    if (concreteOp == nullptr || !concreteOp->hasOneUse() || concreteOp->getNumOperands() != 1) {
        return matchFailed(_log, rewriter, origOp, "ConcreteOp not found or has multiple producers or uses");
    }

    auto preShapeCastOp = concreteOp->getOperand(0).template getDefiningOp<IE::ShapeCastOp>();
    if (preShapeCastOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "Previous ShapeCastOp not found");
    }

    auto newShapeCastOp =
            rewriter.create<IE::ShapeCastOp>(origOp.getLoc(), preShapeCastOp.getResult(), origOp.getShapeAttr());

    mlir::IRMapping mapper;
    mapper.map(concreteOp->getOperand(0), newShapeCastOp.getResult());
    auto newOp = rewriter.clone(*concreteOp, mapper);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::SHAPE);

    rewriter.replaceOp(origOp, newOp->getResults());

    _log.trace("[{0}] Replaced with '{1}'", getDebugName(), concreteOp->getLoc());
    return mlir::success();
}

//
// MoveThroughPoolOp
//

template <class PoolingOp>
class MoveThroughPoolOp final : public mlir::OpRewritePattern<IE::ShapeCastOp> {
public:
    MoveThroughPoolOp(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ShapeCastOp>(ctx), _log(log) {
        this->setDebugName("MoveThroughPoolOp");
    }

private:
    mlir::LogicalResult matchAndRewrite(IE::ShapeCastOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class PoolingOp>
mlir::LogicalResult MoveThroughPoolOp<PoolingOp>::matchAndRewrite(IE::ShapeCastOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    auto poolingOp = origOp.getOperand().getDefiningOp<PoolingOp>();

    if (poolingOp == nullptr || !poolingOp->hasOneUse() || poolingOp->getNumOperands() != 1) {
        return matchFailed(_log, rewriter, origOp, "poolingOp not found or has multiple producers or uses");
    }

    const auto supportedPooling = [&](PoolingOp layerOp) {
        const auto kernels = parseIntArrayAttr<int64_t>(layerOp.getKernelSize());
        const auto padStart = parseIntArrayAttr<int64_t>(layerOp.getPadsBegin());
        const auto padEnd = parseIntArrayAttr<int64_t>(layerOp.getPadsEnd());
        const auto strides = parseIntArrayAttr<int64_t>(layerOp.getStrides());

        mlir::Value input = layerOp.getInput();
        mlir::Value output = layerOp.getOutput();
        auto inputLayout = input.getType().cast<vpux::NDTypeInterface>().getDimsOrder();
        auto outputLayout = output.getType().cast<vpux::NDTypeInterface>().getDimsOrder();
        // input and output layer need to be same
        if (inputLayout != outputLayout) {
            return false;
        }

        auto hasValidKernels = llvm::all_of(kernels, [&](const auto& kernel) {
            return kernel == 1;
        });
        auto hasValidPadStart = llvm::all_of(padStart, [&](const auto& pad) {
            return pad == 0;
        });
        auto hasValidPadEnd = llvm::all_of(padEnd, [&](const auto& pad) {
            return pad == 0;
        });
        auto hasValidStrides = llvm::all_of(strides, [&](const auto& stride) {
            return stride == 1;
        });

        const auto sizeToAlign = std::max(
                VPU::NCEInvariant::getAlignment(input.getType().cast<vpux::NDTypeInterface>().getElementType()),
                VPU::NCEInvariant::getAlignment(output.getType().cast<vpux::NDTypeInterface>().getElementType()));
        const auto origOutShape = getShape(origOp.getResult());

        return hasValidKernels && hasValidPadStart && hasValidPadEnd && hasValidStrides &&
               (origOutShape[Dims4D::Act::C] % sizeToAlign == 0);
    };

    if (!supportedPooling(poolingOp)) {
        return matchFailed(_log, rewriter, origOp, "poolingOp is not eltwise");
    }

    auto preShapeCastOp = poolingOp->getOperand(0).template getDefiningOp<IE::ShapeCastOp>();
    if (preShapeCastOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "Previous ShapeCastOp not found");
    }

    auto newShapeCastOp =
            rewriter.create<IE::ShapeCastOp>(origOp.getLoc(), preShapeCastOp.getResult(), origOp.getShapeAttr());

    mlir::IRMapping mapper;
    mapper.map(poolingOp->getOperand(0), newShapeCastOp.getResult());
    auto newOp = rewriter.clone(*poolingOp, mapper);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::SHAPE);

    rewriter.replaceOp(origOp, newOp->getResults());

    _log.trace("[{0}] Replaced with '{1}'", getDebugName(), poolingOp->getLoc());
    return mlir::success();
}

//
// PropagateShapeCast
//

class PropagateShapeCast final : public IE::PropagateShapeCastBase<PropagateShapeCast> {
public:
    explicit PropagateShapeCast(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

/**
 * ShapeCast operations are insert before and after ops when they call adjust passes, such as GroupConv and Eltwise.
 * And ShapeCast blocked the link relationship for original adjacent ops. Sometimes we can directly move some ops post
 * ShapeCast to reconstruct their link relationship. Such as
 *
 *          ShapeCast                            ShapeCast
 *              |                                    |
 *             Op                                ShapeCast
 *              |                --->                |
 *          ShapeCast                               Op
 *              |                                    |
 *            NceOp                                NceOp
 */
void PropagateShapeCast::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MoveThroughActivationOp<IE::AbsOp>>(&ctx, _log);
    patterns.add<MoveThroughActivationOp<IE::GeluOp>>(&ctx, _log);
    patterns.add<MoveThroughActivationOp<IE::SwishOp>>(&ctx, _log);
    patterns.add<MoveThroughActivationOp<IE::HSwishOp>>(&ctx, _log);
    patterns.add<MoveThroughActivationOp<IE::SigmoidOp>>(&ctx, _log);
    patterns.add<MoveThroughActivationOp<IE::TanhOp>>(&ctx, _log);
    patterns.add<MoveThroughPoolOp<IE::AvgPoolOp>>(&ctx, _log);
    patterns.add<MoveThroughPoolOp<IE::MaxPoolOp>>(&ctx, _log);
    IE::ShapeCastOp::getCanonicalizationPatterns(patterns, &ctx);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createPropagateShapeCastPass(Logger log) {
    return std::make_unique<PropagateShapeCast>(log);
}
