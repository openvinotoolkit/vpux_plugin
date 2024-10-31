//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPURT/utils/barrier_legalization_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"

using namespace vpux;

namespace {
bool isSuitableDmaTask(VPURT::TaskOp taskOp) {
    if (taskOp.getExecutorKind() != VPU::ExecutorKind::DMA_NN) {
        return false;
    }
    auto dma = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(taskOp.getInnerTaskOp());
    VPUX_THROW_WHEN(dma == nullptr, "Invalid inner task type at '{0}'", taskOp->getLoc());
    auto port = dma.getPortVal();
    if (port != 0) {
        return false;
    }
    return dma.getInput().getType().dyn_cast<vpux::NDTypeInterface>().getMemoryKind() == VPU::MemoryKind::DDR;
}

VPURT::TaskOp getFirstSuitableDMAOp(mlir::func::FuncOp funcOp) {
    auto taskOps = funcOp.getOps<VPURT::TaskOp>();
    VPURT::TaskOp dmaOp = nullptr;
    auto iter = llvm::find_if(taskOps, [&](const auto& taskOp) {
        return isSuitableDmaTask(taskOp);
    });
    if (iter != taskOps.end()) {
        dmaOp = *iter;
    }
    return dmaOp;
}

// suitable start barrier cannot be consumed by any other task other than DMA
bool checkSuitableStartBarrier(VPURT::DeclareVirtualBarrierOp barrierOp) {
    for (auto user : barrierOp.getBarrier().getUsers()) {
        auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(user);
        if (llvm::none_of(taskOp.getWaitBarriers(), [=](mlir::Value v) {
                return v.getDefiningOp<VPURT::DeclareVirtualBarrierOp>() == barrierOp;
            })) {
            continue;
        }
        if (taskOp.getExecutorKind() != VPU::ExecutorKind::DMA_NN) {
            return false;
        }
    }
    return true;
}
/*
Check DMA->Bar->DMA pattern with port 0 and DDR channel type
*/
bool checkPattern(VPURT::TaskOp firstDMAOp) {
    if (firstDMAOp == nullptr) {
        return false;
    }
    auto updateBarriers = firstDMAOp.getUpdateBarriers();
    for (const auto& barrier : updateBarriers) {
        auto barrierOp = barrier.getDefiningOp<VPURT::DeclareVirtualBarrierOp>();
        if (checkSuitableStartBarrier(barrierOp)) {
            for (auto user : barrierOp.getBarrier().getUsers()) {
                auto taskOp = mlir::dyn_cast<VPURT::TaskOp>(user);
                if (isSuitableDmaTask(taskOp)) {
                    return true;
                }
            }
        }
    }
    return false;
}

VPURT::TaskOp createSyncDMA(mlir::OpBuilder& builder, mlir::Operation* insertPoint, mlir::Value input,
                            mlir::Value outputBuf, mlir::ValueRange updateBarriers) {
    auto ctx = builder.getContext();
    auto syncDmaLoc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "sync_dma"));
    auto zeroAttr = vpux::getIntAttr(ctx, 0);

    VPUX_THROW_WHEN(insertPoint == nullptr, "Invaild insert point");
    builder.setInsertionPoint(insertPoint);
    auto syncDMATask = VPURT::wrapIntoTaskOp<VPUIP::SyncDMAOp>(
            builder, {}, updateBarriers, syncDmaLoc, input, outputBuf, /*port*/ zeroAttr,
            /*isOutOfOrder*/ nullptr, /*isCritical*/ nullptr, /*dmaHwpId*/ nullptr,
            /*dmaProfilingMetaData*/ nullptr);
    return syncDMATask->getParentOfType<VPURT::TaskOp>();
}

class AddStartBarrierPass final : public VPUIP::arch40xx::AddStartBarrierBase<AddStartBarrierPass> {
public:
    explicit AddStartBarrierPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AddStartBarrierPass::safeRunOnFunc() {
    auto func = getOperation();
    auto firstDmaOp = getFirstSuitableDMAOp(func);
    if (checkPattern(firstDmaOp)) {
        return;
    }
    // Need create new barrer and sync dma task
    auto insertPoint = &func.getBody().front().front();
    mlir::OpBuilder builder(func);
    builder.setInsertionPoint(insertPoint);
    auto loc = mlir::NameLoc::get(mlir::StringAttr::get(&getContext(), "start_barrier"));
    auto barrierOp = builder.create<VPURT::DeclareVirtualBarrierOp>(loc);
    _log.trace("Add start barrier {0}", barrierOp->getLoc());

    // Create dummy input and output buffer
    auto buffers = func.getOps<VPURT::DeclareBufferOp>();
    VPUX_THROW_WHEN(buffers.empty(), "Can not find DeclareBufferOp");
    auto firstDeclareBufferOp = *buffers.begin();
    auto inBuffer = VPUIP::createDummyBuffer(builder, firstDeclareBufferOp);
    auto outBuffer = VPUIP::createDummyBuffer(builder, firstDeclareBufferOp);

    if (firstDmaOp == nullptr) {
        // Create a SyncDMA op as the first DMA
        auto taskOps = func.getOps<VPURT::TaskOp>();
        VPUX_THROW_WHEN(taskOps.empty(), "Can not find TaskOp");
        auto firstTaskOp = *taskOps.begin();
        firstDmaOp = createSyncDMA(builder, firstTaskOp, inBuffer, outBuffer, {});
    }

    _log.trace("Add sync dma for {0}", firstDmaOp->getLoc());
    createSyncDMA(builder, firstDmaOp, inBuffer, outBuffer, {barrierOp.getBarrier()});
    firstDmaOp.getWaitBarriersMutable().append(barrierOp.getBarrier());

    VPURT::verifyBarrierSlots(func, _log);
}
}  // namespace

//
// createAddStartBarrierPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch40xx::createAddStartBarrierPass(Logger log) {
    return std::make_unique<AddStartBarrierPass>(log);
}
