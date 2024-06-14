//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

namespace {

void removeOutputSparse(VPU::NCEOpInterface nceOp, Logger log) {
    if (!VPU::shouldRemoveOutputSparsity(nceOp)) {
        return;
    }

    auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    const auto outputTensorType = clusteredOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto sparseOutputType = outputTensorType.dyn_cast<VPU::SparseTensorType>();

    auto recursivelyRemoveSparseOutput = [&](VPU::ClusteredOpInterface clusteredOp) -> void {
        clusteredOp->getResult(0).setType(sparseOutputType.getData());
        log.nest().trace("Remove output sparsity for op {0} at {1}", clusteredOp->getName(), clusteredOp->getLoc());

        auto users = to_small_vector(clusteredOp->getUsers());
        while (!users.empty()) {
            auto currentOp = users.back();
            users.pop_back();
            if (mlir::isa_and_nonnull<vpux::VPU::ViewLikeOpInterface>(currentOp)) {
                vpux::inferReturnTypes(currentOp, vpux::InferShapedTypeMode::ALL);
                auto nextOps = to_small_vector(currentOp->getUsers());
                users.insert(users.end(), nextOps.begin(), nextOps.end());
            }
        }
    };

    recursivelyRemoveSparseOutput(clusteredOp);
}

//
// RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass
//
class RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass final :
        public VPU::RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPassBase<
                RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass> {
public:
    explicit RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass(Logger log): _log(log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

//
// safeRunOnFunc
//
void RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass::safeRunOnFunc() {
    auto func = getOperation();

    // TODO: E#106239
    // This pass could remove activation sparsity after strategy manager. With these changes, multi-clustering and
    // tiling are done with the cost of activation sparsity being present while sparsity can be reverted. This can have
    // an impact over the performance. Hopefully in the future we can look into refactoring the strategy manager to also
    // take the decision on whether to enable activation sparsity or not.
    func->walk([&](VPU::NCEOpInterface op) {
        removeOutputSparse(op, _log);
    });
}
}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createRemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass(Logger log) {
    return std::make_unique<RemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass>(log);
}
