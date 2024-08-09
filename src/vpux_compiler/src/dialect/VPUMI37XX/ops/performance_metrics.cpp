//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/performance_metrics.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/utils/performance_metrics.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"

#include <npu_37xx_nnrt.hpp>

using namespace vpux;
using namespace npu37xx;

//
// PerformanceMetrics
//

void vpux::VPUMI37XX::PerformanceMetricsOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    nn_public::VpuPerformanceMetrics perf{};

    auto freqTable = VPU::arch37xx::getFrequencyTable();
    perf.freq_base = freqTable.base;
    perf.freq_step = freqTable.step;
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

size_t vpux::VPUMI37XX::PerformanceMetricsOp::getBinarySize() {
    return sizeof(nn_public::VpuPerformanceMetrics);
}

size_t vpux::VPUMI37XX::PerformanceMetricsOp::getAlignmentRequirements() {
    return alignof(nn_public::VpuPerformanceMetrics);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::PerformanceMetricsOp::getAccessingProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}

vpux::ELFNPU37XX::SectionFlagsAttr vpux::VPUMI37XX::PerformanceMetricsOp::getUserProcs() {
    return (ELFNPU37XX::SectionFlagsAttr::SHF_NONE);
}

vpux::VPURT::BufferSection vpux::VPUMI37XX::PerformanceMetricsOp::getMemorySpace() {
    return vpux::VPURT::BufferSection::DDR;
}
