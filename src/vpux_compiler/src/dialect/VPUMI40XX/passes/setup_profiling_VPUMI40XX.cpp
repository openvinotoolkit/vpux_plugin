//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/IR/Builders.h>

using namespace vpux;

namespace {

//
// SetupProfilingVPUMI40XXPass
//

class SetupProfilingVPUMI40XXPass final : public VPUMI40XX::SetupProfilingVPUMI40XXBase<SetupProfilingVPUMI40XXPass> {
public:
    explicit SetupProfilingVPUMI40XXPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;

    mlir::Value createDmaHwpBase(mlir::OpBuilder builderFunc, VPUIP::ProfilingSectionOp dmaSection) {
        _log.trace("createDmaHwpBase");
        VPUX_THROW_UNLESS((dmaSection.getOffset() % VPUIP::HW_DMA_PROFILING_SIZE_BYTES_40XX) == 0,
                          "Unaligned HWP base");
        VPUX_THROW_UNLESS((dmaSection.getSize() % VPUIP::HW_DMA_PROFILING_SIZE_BYTES_40XX) == 0,
                          "Bad DMA section size");

        const auto ctx = builderFunc.getContext();
        const auto outputType = getMemRefType({dmaSection.getSize() / 4}, getUInt32Type(ctx), DimsOrder::C,
                                              VPURT::BufferSection::ProfilingOutput)
                                        .cast<vpux::NDTypeInterface>();
        const auto profilingOutputType =
                mlir::MemRefType::get(outputType.getShape().raw(), outputType.getElementType());
        auto dmaHwpBase = builderFunc.create<VPURT::DeclareBufferOp>(
                mlir::NameLoc::get(mlir::StringAttr::get(ctx, "dmaHwpBase")), profilingOutputType,
                VPURT::BufferSection::ProfilingOutput, 0, dmaSection.getOffset());
        return dmaHwpBase.getResult();
    }

    // Note on 40xx DMA Scratch buffer is mandatory for non-profiling blobs, see E#101929
    mlir::Value createDmaHwpScratch(mlir::OpBuilder builderFunc, mlir::ModuleOp moduleOp) {
        _log.trace("createDmaHwpScratch");
        auto dmaProfMem = IE::getDmaProfilingReservedMemory(moduleOp, VPU::MemoryKind::CMX_NN);
        VPUX_THROW_WHEN(dmaProfMem == nullptr, "Missing DMA HWP scratch buffer");
        auto dmaProfMemOffset = dmaProfMem.getOffset();
        VPUX_THROW_WHEN(dmaProfMemOffset == std::nullopt, "No address allocated.");

        const auto ctx = builderFunc.getContext();
        const auto memKind = IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN), 0);
        const auto outputType =
                getMemRefType({VPUIP::HW_DMA_PROFILING_SIZE_BYTES_40XX / 4}, getUInt32Type(ctx), DimsOrder::C, memKind)
                        .cast<vpux::NDTypeInterface>();

        auto dmaHwpScratch = builderFunc.create<VPURT::DeclareBufferOp>(
                mlir::NameLoc::get(mlir::StringAttr::get(ctx, "dmaHwpScratch")), outputType,
                VPURT::BufferSection::CMX_NN, 0, dmaProfMemOffset.value());
        return dmaHwpScratch.getResult();
    }

    void addDmaHwpBase(mlir::OpBuilder builderFunc, mlir::ModuleOp moduleOp, VPUMI40XX::MappedInferenceOp mpi) {
        _log.trace("addDmaHwpBase");
        auto maybeDmaSection = vpux::getProfilingSection(moduleOp, profiling::ExecutorType::DMA_HW);
        mlir::Value dmaHwpBase = nullptr;
        if (maybeDmaSection) {
            dmaHwpBase = createDmaHwpBase(builderFunc, maybeDmaSection.value());
        } else if (VPU::getArch(moduleOp) == VPU::ArchKind::NPU40XX) {
            // DMA scratch buffer is required only on 40XX
            dmaHwpBase = createDmaHwpScratch(builderFunc, moduleOp);
        }
        if (dmaHwpBase != nullptr) {
            auto dmaHwpBaseOperand = mpi.getDmaHwpBaseMutable();
            dmaHwpBaseOperand.assign(dmaHwpBase);
        }
    }

    void addWorkpointCapture(mlir::OpBuilder builderFunc, mlir::ModuleOp moduleOp, VPUMI40XX::MappedInferenceOp mpi) {
        _log.trace("addWorkpointCapture");

        auto maybeCaptureSection = vpux::getProfilingSection(moduleOp, profiling::ExecutorType::WORKPOINT);
        if (!maybeCaptureSection) {
            _log.trace("No workpoint section");
            return;
        }

        const auto ctx = builderFunc.getContext();
        auto captureSection = maybeCaptureSection.value();
        unsigned pllSizeBytes = captureSection.getSize();
        VPUX_THROW_UNLESS(pllSizeBytes == profiling::WORKPOINT_BUFFER_SIZE, "Bad PLL section size: {0}", pllSizeBytes);
        const auto outputType = getMemRefType({pllSizeBytes / 4}, getUInt32Type(ctx), DimsOrder::C,
                                              VPURT::BufferSection::ProfilingOutput)
                                        .cast<vpux::NDTypeInterface>();
        const auto profilingOutputType =
                mlir::MemRefType::get(outputType.getShape().raw(), outputType.getElementType());

        auto workpointBase = builderFunc.create<VPURT::DeclareBufferOp>(
                mlir::NameLoc::get(mlir::StringAttr::get(ctx, "workpointBase")), profilingOutputType,
                VPURT::BufferSection::ProfilingOutput, 0, captureSection.getOffset());
        auto hwpWorkpointCfg = workpointBase.getResult();

        auto hwpWorkpointOperand = mpi.getHwpWorkpointCfgMutable();
        hwpWorkpointOperand.assign(hwpWorkpointCfg);
    }
};

void SetupProfilingVPUMI40XXPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    IE::CNNNetworkOp netOp;
    mlir::func::FuncOp funcOp;
    IE::CNNNetworkOp::getFromModule(moduleOp, netOp, funcOp);
    mlir::OpBuilder builderFunc(&(funcOp.getFunctionBody()));

    auto mpi = VPUMI40XX::getMPI(funcOp);

    // create DMA hardware profiling base ref in MI
    addDmaHwpBase(builderFunc, moduleOp, mpi);

    // create workpoint cfg ref in MI for hardware profiling
    addWorkpointCapture(builderFunc, moduleOp, mpi);
}

}  // namespace

//
// createSetupProfilingVPUMI40XXPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createSetupProfilingVPUMI40XXPass(Logger log) {
    return std::make_unique<SetupProfilingVPUMI40XXPass>(log);
}
