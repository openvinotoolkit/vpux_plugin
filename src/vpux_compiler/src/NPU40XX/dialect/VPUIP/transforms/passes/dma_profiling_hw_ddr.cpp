//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/profiling/common.hpp"

using namespace vpux;

namespace {

//
//  DMATaskProfilingHwDdrPass
//

class DMATaskProfilingHwDdrPass final : public VPUIP::arch40xx::DMATaskProfilingHwDdrBase<DMATaskProfilingHwDdrPass> {
public:
    explicit DMATaskProfilingHwDdrPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void DMATaskProfilingHwDdrPass::safeRunOnModule() {
    auto module = getOperation();
    auto* ctx = module->getContext();

    IE::CNNNetworkOp netOp;
    mlir::func::FuncOp func;
    IE::CNNNetworkOp::getFromModule(module, netOp, func);

    uint32_t dmaHwpId = 0;
    func->walk([&](VPURT::TaskOp taskOp) {
        if (!vpux::isProfiledDmaTask(taskOp)) {
            return mlir::WalkResult::skip();
        }

        auto taskName = stringifyPrimaryLocation(taskOp->getLoc());
        // Skip DMAs which are used for handling profiling. Such DMAs will not be measured.
        // TODO: Do not use taskName
        if (taskName.find(profiling::PROFILING_CMX_2_DDR_OP_NAME) != std::string::npos) {
            return mlir::WalkResult::skip();
        }

        if (dmaHwpId >= VPUIP::HW_DMA_PROFILING_ID_LIMIT - 1) {
            _log.warning("Some DMA task cannot be profiled.");
            _log.info("First task not profilied: '{0}'", taskName);
            return mlir::WalkResult::interrupt();
        }

        ++dmaHwpId;  // Effective ID start at 1

        _log.trace("DMA HW DDR {0} task: '{1}'", dmaHwpId, taskName);
        auto innerOp = taskOp.getInnerTaskOp();
        auto op = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(innerOp);
        vpux::setDmaHwpIdAttribute(ctx, op, dmaHwpId);
        op.setProfilingMetadata(vpux::getDmaProfilingMetaAttr(ctx, dmaHwpId));
        return mlir::WalkResult::advance();
    });

    // Calculate total size of memory required to store profiling data
    uint32_t totalRecords = dmaHwpId + 1;  // add a dummy record
    auto recordSize = VPUIP::HW_DMA_PROFILING_SIZE_BYTES_40XX;
    auto memKindAttr = IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::DDR), 0);

    mlir::MemRefType outputResult =
            getMemRefType(ShapeRef({totalRecords * recordSize}), getUInt8Type(ctx), DimsOrder::C, memKindAttr);

    // Update network output information with new DMA profiling data
    mlir::OpBuilder builder(&func.getBody().front().front());
    auto profilingResult = addNewProfilingOutput(ctx, func, netOp, outputResult, profiling::ExecutorType::DMA_HW);
    auto returnOp = mlir::dyn_cast_or_null<mlir::func::ReturnOp>(func.getBody().front().getTerminator());
    VPUX_THROW_UNLESS(returnOp != nullptr, "No ReturnOp was found");
    builder.setInsertionPoint(returnOp);
    returnOp.getOperandsMutable().append(profilingResult);
}

}  // namespace

//
// createDMATaskProfilingHwDdrPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::arch40xx::createDMATaskProfilingHwDdrPass(Logger log) {
    return std::make_unique<DMATaskProfilingHwDdrPass>(log);
}
