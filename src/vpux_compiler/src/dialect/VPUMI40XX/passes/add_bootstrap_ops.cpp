//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"

using namespace vpux;

namespace {

class AddBootstrapOpsPass : public VPUMI40XX::AddBootstrapOpsBase<AddBootstrapOpsPass> {
public:
    explicit AddBootstrapOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void reindexEnqueueOps(llvm::SmallVector<VPURegMapped::EnqueueOp> enquOps) {
    if (enquOps.size() == 0) {
        return;
    }

    auto ctx = enquOps[0].getContext();
    auto index = [&ctx](auto taskIdx) {
        return VPURegMapped::IndexType::get(ctx, checked_cast<uint32_t>(taskIdx));
    };

    enquOps[0].getResult().setType(index(0));
    enquOps[0].getPreviousTaskIdxMutable().clear();

    for (size_t i = 1; i < enquOps.size(); i++) {
        auto enqu = enquOps[i];
        enqu.getResult().setType(index(i));
        enqu.getPreviousTaskIdxMutable().assign(enquOps[i - 1]);
    }

    return;
}

void AddBootstrapOpsPass::safeRunOnFunc() {
    auto ctx = &(getContext());
    auto netFunc = getOperation();
    auto mpi = VPUMI40XX::getMPI(netFunc);
    auto builder = mlir::OpBuilder(mpi.getOperation());
    int64_t nBarrs = VPUIP::getNumAvailableBarriers(netFunc);

    std::vector<bool> initialized(nBarrs, false);

    int64_t bootstrapID = 0;
    mlir::Value first;
    for (auto op : netFunc.getOps<VPUMI40XX::ConfigureBarrierOp>()) {
        auto trivialIndexType = VPURegMapped::IndexType::get(ctx, checked_cast<uint32_t>(bootstrapID));
        auto pid = op.getId();
        if (initialized[pid])
            continue;

        auto bootsTrapTask = builder.create<VPUMI40XX::BootstrapOp>(op.getLoc(), trivialIndexType, op->getResult(0));
        if (bootstrapID == 0) {
            first = bootsTrapTask;
        }
        if (bootstrapID == nBarrs) {
            break;
        }
        ++bootstrapID;
        initialized[pid] = true;
    }
    if (first) {
        mpi.getBootstrapTasksMutable().assign(first);
        mpi.setBootstrapTasksCountAttr(builder.getI64IntegerAttr(std::min(nBarrs, bootstrapID)));
    }

    auto parentModule = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto tilesCount = IE::getTileExecutor(parentModule).getCount();

    int64_t bootstrapDmaID = 0;
    VPURegMapped::EnqueueOp firstEnqueue = nullptr;
    if (mpi.getWorkItemTasks()) {
        firstEnqueue = mlir::cast<VPURegMapped::EnqueueOp>(mpi.getWorkItemTasks().getDefiningOp());
    }

    for (int64_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        for (int64_t listIdx = 0; listIdx < 2; listIdx++) {
            auto dmaTaskVal = mpi.getListHead(VPURegMapped::TaskType::DMA, tileIdx, listIdx);
            if (!dmaTaskVal) {
                continue;
            }

            auto dmaTask = mlir::cast<VPURegMapped::TaskOpInterface>(dmaTaskVal.getDefiningOp());
            auto hasEnqu = [](VPURegMapped::TaskOpInterface dma) -> bool {
                auto users = dma.getResult().getUsers();
                auto enquIt = llvm::find_if(users, [](mlir::Operation* user) {
                    return mlir::isa<VPURegMapped::EnqueueOp>(user);
                });
                return enquIt != users.end();
            };

            if (!hasEnqu(dmaTask)) {
                auto startDma = dmaTask;
                auto endDma = dmaTask;
                while (auto nextDma = VPUMI40XX::getNextOp(endDma)) {
                    if (!hasEnqu(nextDma)) {
                        endDma = nextDma;
                    } else {
                        break;
                    }
                }
                auto trivialIndexType = VPURegMapped::IndexType::get(ctx, checked_cast<uint32_t>(bootstrapDmaID));
                auto bootstrapEnqueue = builder.create<VPURegMapped::EnqueueOp>(
                        startDma->getLoc(), trivialIndexType, nullptr, nullptr, VPURegMapped::TaskType::DMA,
                        startDma->getResult(0), endDma->getResult(0));
                if (firstEnqueue) {
                    bootstrapEnqueue.getOperation()->moveBefore(
                            mlir::cast<VPURegMapped::EnqueueOp>(firstEnqueue).getOperation());
                }

                bootstrapDmaID++;
            }
        }
    }

    auto enquOps = to_small_vector(netFunc.getOps<VPURegMapped::EnqueueOp>());
    if (!enquOps.empty()) {
        reindexEnqueueOps(enquOps);
        mpi.getWorkItemTasksMutable().assign(enquOps[0].getResult());
        mpi.setWorkItemCount(enquOps.size());
        mpi.setBootsrapWorkItemsCountAttr(builder.getI64IntegerAttr(bootstrapDmaID));
    } else {
        VPUX_THROW("We expect at least one enqueue operation in the function.");
    }
}

}  // namespace

//
// createAddBootstrapOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createAddBootstrapOpsPass(Logger log) {
    return std::make_unique<AddBootstrapOpsPass>(log);
}
