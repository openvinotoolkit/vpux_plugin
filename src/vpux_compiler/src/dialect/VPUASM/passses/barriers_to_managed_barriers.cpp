//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/Transforms/DialectConversion.h>
#include <vector>
#include "vpux/compiler/dialect/VPUASM/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

using namespace vpux;

namespace {

class BarrierRewriter final : public mlir::OpRewritePattern<VPUASM::ConfigureBarrierOp> {
public:
    BarrierRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUASM::ConfigureBarrierOp>(ctx), _log(log) {
        setDebugName("ConfigureBarrier_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::ConfigureBarrierOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult BarrierRewriter::matchAndRewrite(VPUASM::ConfigureBarrierOp origOp,
                                                     mlir::PatternRewriter& rewriter) const {
    rewriter.replaceOpWithNewOp<VPUASM::ManagedBarrierOp>(origOp, origOp.getSymNameAttr(),  // sym_name
                                                          origOp.getTaskIndexAttr(),        // task_index
                                                          origOp.getWorkItemIdxAttr(),      // work_item_idx
                                                          rewriter.getUI32IntegerAttr(0),   // work_item_count
                                                          origOp.getIdAttr(),               // id
                                                          origOp.getNextSameIdAttr(),       // next_same_id
                                                          origOp.getProducerCountAttr(),    // producer_count
                                                          origOp.getConsumerCountAttr());   // consumer_count
    return mlir::success();
}

class BarriersToManagedBarriersPass final :
        public VPUASM::BarriersToManagedBarriersBase<BarriersToManagedBarriersPass> {
public:
    explicit BarriersToManagedBarriersPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void BarriersToManagedBarriersPass::safeRunOnFunc() {
    auto& ctx = getContext();
    mlir::func::FuncOp netFunc = getOperation();

    mlir::ConversionTarget target(ctx);

    target.addIllegalOp<VPUASM::ConfigureBarrierOp>();
    target.addLegalOp<VPUASM::ManagedBarrierOp>();

    mlir::RewritePatternSet patterns(&ctx);

    patterns.add<BarrierRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(netFunc, target, std::move(patterns)))) {
        signalPassFailure();
    }

    return;
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUASM::createBarriersToManagedBarriersPass(Logger log) {
    return std::make_unique<BarriersToManagedBarriersPass>(log);
}
