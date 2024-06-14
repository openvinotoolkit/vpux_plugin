//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/interfaces/barrier_simulator.hpp"

#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

//
// VirtualBarrierRewrite
//

class VirtualBarrierRewrite final : public mlir::OpRewritePattern<VPURT::DeclareVirtualBarrierOp> {
public:
    VirtualBarrierRewrite(mlir::MLIRContext* ctx, const VPURT::BarrierSimulator& barrierSim, Logger log)
            : mlir::OpRewritePattern<VPURT::DeclareVirtualBarrierOp>(ctx), _barrierSim(barrierSim), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPURT::DeclareVirtualBarrierOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    const VPURT::BarrierSimulator& _barrierSim;
    Logger _log;
};

mlir::LogicalResult VirtualBarrierRewrite::matchAndRewrite(VPURT::DeclareVirtualBarrierOp origOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Found DeclareVirtualBarrierOp Operation '{0}'", origOp->getLoc());

    const auto& conf = _barrierSim.getConfig(origOp.getBarrier());
    _log.nest().trace("Use physical barrier ID '{0}'", conf.realId);

    rewriter.replaceOpWithNewOp<VPURT::ConfigureBarrierOp>(origOp, conf.realId, origOp.getIsFinalBarrier());
    return mlir::success();
}

//
// AssignPhysicalBarriersPass
//

class AssignPhysicalBarriersPass final : public VPURT::AssignPhysicalBarriersBase<AssignPhysicalBarriersPass> {
public:
    explicit AssignPhysicalBarriersPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AssignPhysicalBarriersPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    const auto numBarriers =
            numBarriersOpt.hasValue() ? std::optional<int64_t>(numBarriersOpt.getValue()) : std::nullopt;

    auto& barrierSim = getAnalysis<VPURT::BarrierSimulator>();
    if (!barrierSim.isDynamicBarriers()) {
        return;
    }

    if (mlir::failed(barrierSim.checkProducerCount(_log.nest()))) {
        signalPassFailure();
        return;
    }
    if (mlir::failed(barrierSim.checkProducerAndConsumerCount(_log.nest()))) {
        signalPassFailure();
        return;
    }
    if (mlir::failed(barrierSim.simulateBarriers(_log.nest(), numBarriers))) {
        signalPassFailure();
        return;
    }

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<VPURT::DeclareVirtualBarrierOp>();
    target.addLegalOp<VPURT::ConfigureBarrierOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<VirtualBarrierRewrite>(&ctx, barrierSim, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createAssignPhysicalBarriersPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::createAssignPhysicalBarriersPass(Logger log) {
    return std::make_unique<AssignPhysicalBarriersPass>(log);
}
