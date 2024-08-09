//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/make_ops_with_distributed_tensor_strategy_getter.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/overlap_distribution_utils.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;
using namespace VPU;

namespace {

//
// MakeOpsWithDistributedTensorPass
//

class MakeOpsWithDistributedTensorPass final :
        public MakeOpsWithDistributedTensorBase<MakeOpsWithDistributedTensorPass> {
public:
    MakeOpsWithDistributedTensorPass(Logger log): _enableExplicitDistributedTensorAttr(false) {
        Base::initLogger(log, Base::getArgumentName());
    };

    explicit MakeOpsWithDistributedTensorPass(bool enableExplicitDistributedTensorAttr, Logger log)
            : _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    void safeRunOnFunc() final;
    static bool areOpSiblingsNeeded(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy);
};

mlir::LogicalResult MakeOpsWithDistributedTensorPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (enableExplicitDistributedTensorAttr.hasValue()) {
        _enableExplicitDistributedTensorAttr = enableExplicitDistributedTensorAttr.getValue();
        return mlir::success();
    }

    return mlir::success();
}

bool MakeOpsWithDistributedTensorPass::areOpSiblingsNeeded(VPU::ClusteredOpInterface clusteredOp,
                                                           VPU::MultiClusterStrategy strategy) {
    if (!outputOverlappedParamsIsHaloSupported(clusteredOp)) {
        return false;
    }
    const auto outputTensorDistributionMode = vpux::VPU::getOutputTensorDistributionMode(clusteredOp, strategy);
    if (outputTensorDistributionMode == DistributionMode::OVERLAPPED &&
        !mlir::isa<VPU::SWOpInterface>(clusteredOp.getOperation())) {
        return true;
    }

    return false;
}

//
// safeRunOnModule
//

void MakeOpsWithDistributedTensorPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    llvm::DenseMap<mlir::OpResult, OverlapDistributionParams> overlapParamsLookup;
    std::vector<std::set<VPU::ClusteredOpInterface>> siblingGroups;

    auto findInSiblingGroups = [&](mlir::Operation* consumerOp) -> std::optional<std::set<VPU::ClusteredOpInterface>> {
        auto clusteredConsumerOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(consumerOp);
        if (clusteredConsumerOp == nullptr) {
            return std::nullopt;
        }
        for (const auto& siblingGroup : siblingGroups) {
            if (siblingGroup.find(clusteredConsumerOp) != siblingGroup.end()) {
                return siblingGroup;
            }
        }
        return std::nullopt;
    };

    func->walk([&](VPU::ClusteredOpInterface clusteredOp) {
        std::set<VPU::ClusteredOpInterface> opSiblings = {};
        const auto strategyAttr = clusteredOp.getMultiClusterStrategy();
        if (!strategyAttr.has_value()) {
            return;
        }
        auto strategy = strategyAttr.value();

        if (areOpSiblingsNeeded(clusteredOp, strategy)) {
            mlir::Operation* consumerOp = nullptr;
            if (isPassthroughOp(clusteredOp.getOperation())) {
                // For passthrough ops, ensure input and output tensors use the same pool of ops to determine the
                // distribution
                consumerOp = clusteredOp.getOperation();
            } else {
                for (const auto& consumer : clusteredOp->getUsers()) {
                    // find first valid consumer and use it to get all its clustered siblings
                    if (mlir::isa<VPU::ClusteredOpInterface>(consumer) || isPassthroughOp(consumer)) {
                        consumerOp = consumer;
                        break;
                    }
                }
            }
            if (consumerOp != nullptr) {
                auto cachedSiblingGroup = findInSiblingGroups(consumerOp);
                if (cachedSiblingGroup.has_value()) {
                    opSiblings.merge(cachedSiblingGroup.value());
                }
                if (opSiblings.empty()) {
                    opSiblings = getSiblingOps(consumerOp);
                    if (!opSiblings.empty()) {
                        auto clusteredSiblingOp = *opSiblings.begin();
                        // The cache may be missed when consumer is not a clustered op, so we need to check it again
                        // here and store it if necessary
                        if (!findInSiblingGroups(clusteredSiblingOp.getOperation()).has_value()) {
                            siblingGroups.emplace_back(opSiblings);
                        }
                    }
                }
            }
        }

        for (const auto& opResult : clusteredOp->getResults()) {
            overlapParamsLookup.insert(std::make_pair(
                    opResult,
                    getOverlapDistributionParams(clusteredOp, opResult.getType().cast<vpux::NDTypeInterface>(),
                                                 opSiblings, strategy)));
        }
    });

    mlir::RewritePatternSet patterns(&ctx);
    auto strategy = vpux::VPU::createMakeOpsWithDistributedTensorStrategyGetter(func, overlapParamsLookup,
                                                                                _enableExplicitDistributedTensorAttr);
    // Both ACT Shaves and DPUs are grouped together in NCE clusters, in a symmetric manner.
    // Each NCE cluster has the same amount of DPUs and ACT shaves.
    // Thus shaves have the availability for distributing across clusters similar to DPUs.
    strategy->addPatterns(patterns, _log);

    mlir::ConversionTarget target(ctx);

    target.markUnknownOpDynamicallyLegal([&](mlir::Operation* op) {
        if (auto clusteredOp = mlir::dyn_cast<ClusteredOpInterface>(op)) {
            if (op->hasAttr(multiClusterStrategy))
                return false;
        }

        return true;
    });

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMakeOpsWithDistributedTensorPass
//

std::unique_ptr<mlir::Pass> VPU::createMakeOpsWithDistributedTensorPass(bool enableExplicitDistributedTensorAttr,
                                                                        Logger log) {
    return std::make_unique<MakeOpsWithDistributedTensorPass>(enableExplicitDistributedTensorAttr, log);
}
