//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/strategy_manager.hpp"

using namespace vpux;
using namespace VPU;

namespace {

//
// MultiClusterStrategyAssignmentPass
//

class MultiClusterStrategyAssignmentPass final :
        public MultiClusterStrategyAssignmentBase<MultiClusterStrategyAssignmentPass> {
public:
    explicit MultiClusterStrategyAssignmentPass(bool enablePrefetchTiling, Logger log)
            : _enablePrefetchTiling(enablePrefetchTiling) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    mlir::LogicalResult initializeOptions(StringRef options) final;
    void safeRunOnFunc() final;

private:
    bool _enablePrefetchTiling = true;
};

mlir::LogicalResult MultiClusterStrategyAssignmentPass::initializeOptions(StringRef options) {
    if (mlir::failed(Base::initializeOptions(options))) {
        return mlir::failure();
    }
    if (tilingMode.hasValue()) {
        _log.trace("Overloading enablePrefetchTiling with an MLIR variable");
        _enablePrefetchTiling = tilingMode.getValue() == "PREFETCH";
    }
    return mlir::success();
}

//
// safeRunOnFunc
//

void MultiClusterStrategyAssignmentPass::safeRunOnFunc() {
    auto func = getOperation();

    auto module = func->getParentOfType<mlir::ModuleOp>();

    auto tileOp = IE::getTileExecutor(module);
    VPUX_THROW_UNLESS(tileOp != nullptr, "Failed to get NCE_Cluster information");

    if (tileOp.getCount() > 1) {
        StrategyManager strategyManager(func, tileOp.getCount(), _enablePrefetchTiling, _log.nest());
        _log.trace("Greedy Strategy Assignment");
        auto module = func->getParentOfType<mlir::ModuleOp>();
        auto enableMultiClusterForSWLayer = IE::getAvailableExecutor(module, VPU::ExecutorKind::SHAVE_ACT) != nullptr;
        strategyManager.assignMultiClusterStrategy(enableMultiClusterForSWLayer);

        _log.trace("Execute Subgraph Optimization");
        strategyManager.optimizeMulticlusterStrategy();
        _log.trace("Remove Temporary Strategy");
        strategyManager.removeTemporaryMulticlusterStrategy();
    }
}

}  // namespace

//
// createMultiClusterStrategyAssignmentPass
//

std::unique_ptr<mlir::Pass> VPU::createMultiClusterStrategyAssignmentPass(bool enablePrefetchTiling, Logger log) {
    return std::make_unique<MultiClusterStrategyAssignmentPass>(enablePrefetchTiling, log);
}
