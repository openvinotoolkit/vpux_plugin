//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

class AddUpdateBarrierForSwKernelsPass final :
        public VPURT::arch37xx::AddUpdateBarrierForSwKernelsBase<AddUpdateBarrierForSwKernelsPass> {
public:
    explicit AddUpdateBarrierForSwKernelsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AddUpdateBarrierForSwKernelsPass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](VPUIP::SwKernelOp origOp) {
        mlir::OpBuilder builder(origOp);
        OpBuilderLogger builderLog(_log);

        auto origTaskOp = origOp->getParentOfType<VPURT::TaskOp>();
        auto origTaskOpUpdateBarriers = origTaskOp.getUpdateBarriers();
        if (!origTaskOpUpdateBarriers.empty()) {
            return;
        }

        builder.setInsertionPoint(origTaskOp);

        auto loc = origOp.getLoc();
        auto newUpdateBarrier = builder.create<VPURT::DeclareVirtualBarrierOp>(loc).getBarrier();
        origTaskOp.getUpdateBarriersMutable().append(newUpdateBarrier);
    });
    VPURT::postProcessBarrierOps(func);
    VPURT::verifyBarrierSlots(func, _log);
}
}  // namespace

//
// createAddUpdateBarrierForSwKernelsPass
//

std::unique_ptr<mlir::Pass> vpux::VPURT::arch37xx::createAddUpdateBarrierForSwKernelsPass(Logger log) {
    return std::make_unique<AddUpdateBarrierForSwKernelsPass>(log);
}
