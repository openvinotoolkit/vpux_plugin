//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux_elf/writer.hpp>
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/utils/performance_metrics.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;

void vpux::ELF::PerformanceMetricsOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    nn_public::VpuPerformanceMetrics perf{};

    perf.freq_base = VPU::getFreqBase(VPU::ArchKind::NPU40XX);
    perf.freq_step = VPU::getFreqStep(VPU::ArchKind::NPU40XX);
    perf.bw_base = VPU::getBWBase();
    perf.bw_step = VPU::getBWStep();

    auto operation = getOperation();
    auto mainModule = operation->getParentOfType<mlir::ModuleOp>();
    auto memRes = IE::getUsedMemory(mainModule);
    // Here we must get AF from NCE res (a TileResourceOp) as the AF attribute is attached to tile op
    mainModule.walk([&](IE::TileResourceOp res) {
        const auto execKind = VPU::getKindValue<VPU::ExecutorKind>(res);
        if (VPU::ExecutorKind::NCE == execKind) {
            perf.activity_factor = VPU::getActivityFactor(execKind, mainModule, res);
            VPUX_THROW_WHEN(perf.activity_factor == VPU::INVALID_AF, "Invalid activity factor!");
        }
    });

    auto numEntries = VPU::getNumEntries();
    auto byBWScales = VPU::getBWScales();
    auto byBWTicks = VPU::getBWTicks(mainModule);
    for (size_t row = 0; row < numEntries; ++row) {
        for (size_t column = 0; column < numEntries; ++column) {
            perf.scalability[row][column] = byBWScales[column];
            perf.ticks[row][column] = byBWTicks[row][column];
        }
    }

    const auto ptrCharTmp = reinterpret_cast<uint8_t*>(&perf);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::ELF::PerformanceMetricsOp::getBinarySize() {
    return sizeof(nn_public::VpuPerformanceMetrics);
}

size_t vpux::ELF::PerformanceMetricsOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuPerformanceMetrics);
}

vpux::ELF::SectionFlagsAttr vpux::ELF::PerformanceMetricsOp::getAccessingProcs(mlir::SymbolUserMap&) {
    return ELF::SectionFlagsAttr::SHF_NONE;
}

vpux::ELF::SectionFlagsAttr vpux::ELF::PerformanceMetricsOp::getUserProcs() {
    return ELF::SectionFlagsAttr::SHF_NONE;
}

std::optional<ELF::SectionSignature> vpux::ELF::PerformanceMetricsOp::getSectionSignature() {
    return ELF::SectionSignature(vpux::ELF::generateSignature("perf", "metrics"), ELF::SectionFlagsAttr::SHF_NONE,
                                 ELF::SectionTypeAttr::VPU_SHT_PERF_METRICS);
}

bool vpux::ELF::PerformanceMetricsOp::hasMemoryFootprint() {
    return true;
}

void vpux::ELF::PerformanceMetricsOp::build(mlir::OpBuilder& builder, mlir::OperationState& state) {
    build(builder, state, "PerfMetrics");
}
