//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"
#include "vpux/compiler/act_kernels/shave_binary_resources.h"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/utils/core/mem_size.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace npu40xx;

namespace vpux {
namespace NPUReg40XX {

constexpr uint32_t defaultActRtCodeSectionSize = static_cast<uint32_t>((1_MB).to<vpux::Byte>().count());
constexpr uint64_t defaultActRtEntry = 0x1C000000;

// ActShave profiling metric mask
//  [0]  BIT_STALL_CYCLE_CNT_EN
//  [1]  BIT_EXEC_INST_CNT_EN
//  [2]  BIT_CLK_CYCLE_CNT_EN
//  [3]  BIT_BRANCH_TAKEN_CNT_EN
//  [4]  BIT_INST_BRKP0_CNT_EN
//  [5]  BIT_INST_BRKP1_CNT_EN
//  [6]  BIT_DATA_BRKP0_CNT_EN
//  [7]  BIT_DATA_BRKP1_CNT_EN
//  [8]  BIT_GO_COUNT_EN
//  [9]  BIT_LSU0_RBYTE_CNT_EN
//  [10] BIT_LSU0_WBYTE_CNT_EN
//  [11] BIT_LSU1_RBYTE_CNT_EN
//  [12] BIT_LSU1_WBYTE_CNT_EN
//  Stall count instructions:
//  [16] SWIH
//  [17] Other interrupts
//  [18] LSU0 Stall (waiting for data)
//  [19] LSU1 Stall (waiting for data)
//  [20] LSU0 Access Stall
//  [21] LSU1 Access Stall
//  [22] Instruction buffer Low Stall
//  [23] Discontinuity Fetch Stall
//  [24] Discontinuity Decode Stall (too much data in instruction buffer at end of delay slots)
//  [25] Discontinuity Starve Stall
//  [26] Instruction buffer Low during discontinuity
//
//  [27] FRC_DURATION_EN
//  [28] FRC_TIMESTAMP_EN
constexpr uint32_t defaultPerfMetricsMask = 0x183C000F;
}  // namespace NPUReg40XX
}  // namespace vpux

//
// MappedInferenceOp
//

void vpux::NPUReg40XX::MappedInferenceOp::serialize(elf::writer::BinaryDataSection<uint8_t>& binDataSection) {
    auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
    bool isActShaveProfilingEnabled =
            vpux::getProfilingSection(moduleOp, profiling::ExecutorType::ACTSHAVE).has_value();

    nn_public::VpuMappedInference mi{};

    mi.vpu_nnrt_api_ver = VPU_NNRT_40XX_API_VER;

    mi.barrier_configs.count = getBarrierCount();
    mi.media_tasks.count = getMediaCount();

    auto dmaDDRCountVec = parseIntArrayAttr<int64_t>(getDmaDDRCount());
    size_t totalDDRDmaCount = 0;
    VPUX_THROW_WHEN(dmaDDRCountVec.size() > nn_public::VPU_MAX_DMA_ENGINES, "Too many DMA DDR lists");
    for (size_t listIdx = 0; listIdx < dmaDDRCountVec.size(); ++listIdx) {
        mi.dma_tasks_ddr_[listIdx].count = dmaDDRCountVec[listIdx];
        totalDDRDmaCount += mi.dma_tasks_ddr_[listIdx].count;
    }

    auto dmaCMXCountVec = parseIntArrayAttr<int64_t>(getDmaCMXCount());
    size_t totalCMXDmaCount = 0;
    VPUX_THROW_WHEN(dmaCMXCountVec.size() > nn_public::VPU_MAX_DMA_ENGINES, "Too many DMA CMX lists");
    for (size_t listIdx = 0; listIdx < dmaCMXCountVec.size(); ++listIdx) {
        mi.dma_tasks_cmx_[listIdx].count = dmaCMXCountVec[listIdx];
        totalCMXDmaCount += mi.dma_tasks_cmx_[listIdx].count;
    }

    auto invariantCountVec = parseIntArrayAttr<int64_t>(getInvariantCount());
    VPUX_THROW_WHEN(invariantCountVec.size() > nn_public::VPU_MAX_TILES, "Too many Invariant lists");
    for (size_t listIdx = 0; listIdx < invariantCountVec.size(); ++listIdx) {
        mi.invariants[listIdx].count = invariantCountVec[listIdx];
    }

    auto variantCountVec = parseIntArrayAttr<int64_t>(getVariantCount());
    VPUX_THROW_WHEN(variantCountVec.size() > nn_public::VPU_MAX_TILES, "Too many Variant lists");
    for (size_t listIdx = 0; listIdx < variantCountVec.size(); ++listIdx) {
        mi.variants[listIdx].count = variantCountVec[listIdx];
    }

    auto actKernelRangesCountVec = parseIntArrayAttr<int64_t>(getActKernelRangesCount());
    VPUX_THROW_WHEN(actKernelRangesCountVec.size() > nn_public::VPU_MAX_TILES, "Too many ActKernelRange lists");
    for (size_t listIdx = 0; listIdx < actKernelRangesCountVec.size(); ++listIdx) {
        mi.act_kernel_ranges[listIdx].count = actKernelRangesCountVec[listIdx];
    }

    auto actKernelInvocationsCountVec = parseIntArrayAttr<int64_t>(getActKernelInvocationsCount());
    VPUX_THROW_WHEN(actKernelInvocationsCountVec.size() > nn_public::VPU_MAX_TILES, "Too many ActKernelInvo lists");
    for (size_t listIdx = 0; listIdx < actKernelInvocationsCountVec.size(); ++listIdx) {
        mi.act_kernel_invocations[listIdx].count = actKernelInvocationsCountVec[listIdx];
    }

    mi.shv_rt_configs.dpu_perf_mode = nn_public::VpuHWPStatMode::MODE3;
    if (getActKernelInvocationsCount()) {
        mi.shv_rt_configs.use_schedule_embedded_rt = false;
        mi.shv_rt_configs.code_window_buffer_size = NPUReg40XX::defaultActRtCodeSectionSize;
        // TODO: E-74314 nnActEntry.40xx.elf has a .versiondata section that contains a single uint32_t
        // This should be read and set in mi.shvRtConfigs_.runtimeVersion_
        mi.shv_rt_configs.runtime_version = 0;
        mi.shv_rt_configs.runtime_entry = NPUReg40XX::defaultActRtEntry;
        mi.shv_rt_configs.perf_metrics_mask = isActShaveProfilingEnabled ? NPUReg40XX::defaultPerfMetricsMask : 0;

        if (getActShaveRt().has_value()) {
            auto actShaveRtRef = mlir::SymbolTable::lookupNearestSymbolFrom(getOperation(), getActShaveRt().value());
            auto actShaveRtOp = mlir::cast<NPUReg40XX::ActShaveRtOp>(actShaveRtRef);

            mi.shv_rt_configs.use_schedule_embedded_rt = true;
            mi.shv_rt_configs.code_window_buffer_size = checked_cast<uint32_t>(actShaveRtOp.getBinarySize());
            mi.shv_rt_configs.runtime_entry = actShaveRtOp.getKernelEntry();
            mi.shv_rt_configs.runtime_version = actShaveRtOp.getVersion();
        }
    }

    if (getManagedMappedInference().has_value()) {
        mi.managed_inference.count = 1;
    }

    // Look only at the DMA tasks belonging to the first (and only) DMA engine
    std::tie(mi.task_storage_counts_.dma_ddr_count, mi.task_storage_counts_.dma_cmx_count) =
            VPUMI40XX::compute_dma_split(totalDDRDmaCount, totalCMXDmaCount);
    mi.task_storage_counts_.dpu_invariant_count = nn_public::VPU_INVARIANT_COUNT;
    mi.task_storage_counts_.dpu_variant_count = nn_public::VPU_VARIANT_COUNT;
    mi.task_storage_counts_.act_range_count = nn_public::VPU_KERNEL_RANGE_COUNT;
    mi.task_storage_counts_.act_invo_count = nn_public::VPU_KERNEL_INVO_COUNT;
    mi.task_storage_counts_.media_count = nn_public::VPU_MEDIA_COUNT;

    uint8_t* ptrCharTmp = reinterpret_cast<uint8_t*>(&mi);
    binDataSection.appendData(ptrCharTmp, getBinarySize());
}

size_t vpux::NPUReg40XX::MappedInferenceOp::getBinarySize() {
    return sizeof(nn_public::VpuMappedInference);
}
