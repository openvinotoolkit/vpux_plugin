//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/workload_splitter_base.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/sparsity_constraint.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

using namespace vpux;
using namespace VPU;

vpux::VPU::WorkloadSplitter37XX::WorkloadSplitter37XX(mlir::func::FuncOp funcOp, vpux::Logger log)
        : WorkloadSplitterBase(funcOp, VPU::supportedChannelsDW, log) {
}

SmallVector<Shape> vpux::VPU::WorkloadSplitter37XX::getPerClusterOffsetsCorrection(VPU::NCEOpInterface nceOp) {
    auto nceClusterTilingOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(nceOp->getParentOp());

    if (nceClusterTilingOp == nullptr) {
        return {};
    }

    auto outputType = nceClusterTilingOp->getResult(0).getType();
    auto distributedOut = outputType.dyn_cast<VPU::DistributedTensorType>();
    if (distributedOut == nullptr) {
        return {};
    }

    // TODO: E#73931
    // PermuteQuantize output will always have memory and compute equal for now.
    return distributedOut.getPerClusterMemoryShapeOffsets();
}

bool vpux::VPU::WorkloadSplitter37XX::isNCEPermuteOffsetsCorrectionNeeded(VPU::NCEOpInterface nceOp) {
    auto nceClusterTilingOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(nceOp->getParentOp());

    if (nceClusterTilingOp == nullptr) {
        return false;
    }

    auto outputType = nceClusterTilingOp->getResult(0).getType();

    return outputType.isa<VPU::DistributedTensorType>();
}

namespace {

//
// CorrectNCEWorkloads
//

class CorrectNCEWorkloadsPass final : public VPU::arch37xx::CorrectNCEWorkloadsBase<CorrectNCEWorkloadsPass> {
public:
    explicit CorrectNCEWorkloadsPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void CorrectNCEWorkloadsPass::safeRunOnFunc() {
    auto func = getOperation();
    const auto arch = getArch(func);
    auto sparsityConstraint = VPU::getSparsityConstraint(arch);

    WorkloadSplitter37XX splitter(func, _log);
    splitter.correctInvalidWorkload(sparsityConstraint);
}

}  // namespace

//
// createCorrectNCEWorkloadsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::arch37xx::createCorrectNCEWorkloadsPass(Logger log) {
    return std::make_unique<CorrectNCEWorkloadsPass>(log);
}
