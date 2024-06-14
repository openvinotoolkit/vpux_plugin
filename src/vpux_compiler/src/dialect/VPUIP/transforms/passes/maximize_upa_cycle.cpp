//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

using namespace vpux;

namespace {

class MaximizeUPACyclesPass final : public VPUIP::MaximizeUPACyclesBase<MaximizeUPACyclesPass> {
public:
    explicit MaximizeUPACyclesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

class CycleComparator {
    AsyncDepsInfo& _depsInfo;
    std::function<size_t(mlir::async::ExecuteOp)> _cycleGetter;

public:
    CycleComparator(AsyncDepsInfo& depsInfo, std::function<size_t(mlir::async::ExecuteOp)> func)
            : _depsInfo(depsInfo), _cycleGetter(std::move(func)) {
    }
    bool operator()(size_t opIndex1, size_t opIndex2) const {
        auto execOp1 = _depsInfo.getExecuteOpAtIndex(opIndex1);
        auto execOp2 = _depsInfo.getExecuteOpAtIndex(opIndex2);

        return _cycleGetter(execOp1) < _cycleGetter(execOp2);
    }
};

void recalculateUPACycles(mlir::func::FuncOp func, AsyncDepsInfo& depsInfo, mlir::async::ExecuteOp execOp) {
    const auto execOpIdx = depsInfo.getIndex(execOp);

    SmallVector<size_t> deps = {};
    for (auto dep : depsInfo.getOpDeps(execOpIdx)) {
        deps.push_back(static_cast<size_t>(dep));
    }
    SmallVector<size_t> consumers = {};
    for (auto con : depsInfo.getConsumerOps(execOpIdx)) {
        consumers.push_back(static_cast<size_t>(con));
    }

    size_t upaCycleBegin = 0;
    size_t upaCycleEnd = getAsyncExecuteCycleEnd(to_small_vector(func.getOps<mlir::async::ExecuteOp>()).back());

    if (!deps.empty()) {
        const auto element =
                *std::max_element(deps.begin(), deps.end(), CycleComparator(depsInfo, getAsyncExecuteCycleEnd));
        upaCycleBegin = getAsyncExecuteCycleEnd(depsInfo.getExecuteOpAtIndex(element));
    }

    if (!consumers.empty()) {
        const auto element = *std::min_element(consumers.begin(), consumers.end(),
                                               CycleComparator(depsInfo, getAsyncExecuteCycleBegin));
        upaCycleEnd = getAsyncExecuteCycleBegin(depsInfo.getExecuteOpAtIndex(element));
    }

    execOp->setAttr(cycleBegin, getIntAttr(execOp->getContext(), upaCycleBegin));
    execOp->setAttr(cycleEnd, getIntAttr(execOp->getContext(), upaCycleEnd));
    execOp->setAttr(cycleCostAttrName, getIntAttr(execOp->getContext(), upaCycleEnd - upaCycleBegin));
}

void MaximizeUPACyclesPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& depsInfo = getAnalysis<AsyncDepsInfo>();
    depsInfo.buildConsMap();

    func->walk([&](mlir::async::ExecuteOp asyncExec) {
        // recalculate cycle for UPA according to its dependencies and consumers
        if (!asyncExec->hasAttr(VPUIP::VPUIPDialect::getExecutorAttrName())) {
            return;
        }

        if (VPUIP::VPUIPDialect::getExecutorKind(asyncExec) != VPU::ExecutorKind::SHAVE_UPA) {
            // skip non target executor operations
            return;
        }

        // calculate UPA cycles
        recalculateUPACycles(func, depsInfo, asyncExec);
    });
}

}  // namespace

//
// createMaximizeUPACyclesPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createMaximizeUPACyclesPass(Logger log) {
    return std::make_unique<MaximizeUPACyclesPass>(log);
}
