//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/wrap_vpu_ops_in_ncecluster_tiling_strategy_getter.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;
using namespace VPU;

namespace {

//
// WrapVPUOpsInNCEClusterTilingPass
//

class WrapVPUOpsInNCEClusterTilingPass final :
        public WrapVPUOpsInNCEClusterTilingBase<WrapVPUOpsInNCEClusterTilingPass> {
public:
    WrapVPUOpsInNCEClusterTilingPass(Logger log): _enableExplicitDistributedTensorAttr(false) {
        Base::initLogger(log, Base::getArgumentName());
    };

    explicit WrapVPUOpsInNCEClusterTilingPass(bool enableExplicitDistributedTensorAttr, Logger log)
            : _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    void safeRunOnFunc() final;
};

mlir::LogicalResult WrapVPUOpsInNCEClusterTilingPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (enableExplicitDistributedTensorAttr.hasValue()) {
        _enableExplicitDistributedTensorAttr = enableExplicitDistributedTensorAttr.getValue();
        return mlir::success();
    }

    return mlir::success();
}

//
// safeRunOnModule
//

void WrapVPUOpsInNCEClusterTilingPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);

    auto strategy =
            vpux::VPU::createWrapVPUOpsInNCEClusterTilingStrategyGetter(func, _enableExplicitDistributedTensorAttr);
    // Both ACT Shaves and DPUs are grouped together in NCE clusters, in a symmetric manner.
    // Each NCE cluster has the same amount of DPUs and ACT shaves.
    // Thus shaves have the availability for distributing across clusters similar to DPUs.
    strategy->addPatterns(patterns, _log);

    mlir::ConversionTarget target(ctx);

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (auto clusteredOp = mlir::dyn_cast<ClusteredOpInterface>(op)) {
            auto strategy = clusteredOp.getMultiClusterStrategy();
            if (strategy.has_value()) {
                return (op->getParentOfType<NCEClusterTilingOp>() != nullptr);
            }
        }

        return true;
    });

    target.addLegalOp<NCEClusterTilingOp>();

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createWrapVPUOpsInNCEClusterTilingPass
//

std::unique_ptr<mlir::Pass> VPU::createWrapVPUOpsInNCEClusterTilingPass(bool enableExplicitDistributedTensorAttr,
                                                                        Logger log) {
    return std::make_unique<WrapVPUOpsInNCEClusterTilingPass>(enableExplicitDistributedTensorAttr, log);
}
