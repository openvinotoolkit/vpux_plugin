//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUIPDPU2NPUReg40XX/dpu_variant_rewriter.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/utils.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/lower_to_registers.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"

using namespace vpux;
using namespace vpux::VPURegMapped;

namespace vpux {
namespace vpuipdpu2npureg40xx {

DPUVariantRewriter::DPUVariantRewriter(mlir::MLIRContext* ctx, Logger log, VPU::DPUDryRunMode dryRunMode)
        : mlir::OpRewritePattern<VPUIPDPU::DPUVariantOp>(ctx, mlir::PatternBenefit(2)),
          _log(log),
          _dryRunMode(dryRunMode) {
    setDebugName("DPUVariant_VPUASM2NPUReg40XXRewriter");
}

mlir::LogicalResult DPUVariantRewriter::matchAndRewrite(VPUIPDPU::DPUVariantOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto initValues = NPUReg40XX::RegMapped_DpuVariantRegisterType::getResetInitilizationValues();

    // index incremented by one by runtime logic. Something to do with preemption
    // This value can be change if needed in the future. For now we use the index+1 just because it is
    // convinient for when we want to preempt when running on Simics (E71635)
    const uint64_t maxTaskId = (1ull << NPUReg40XX::RegField_var_tagType::getRegFieldWidth()) - 1;
    auto taskIdx = checked_cast_reg<NPUReg40XX::RegField_var_tagType>(
            static_cast<uint64_t>(origOp.getTaskIndex().getValue() % maxTaskId + 1));

    VPURegMapped::updateRegMappedInitializationValues(initValues, {
                                                                          {"invar_ptr", {{"var_tag", taskIdx}}},
                                                                          {"dpu_cfg",
                                                                           {{"workload_start_odu", 0b1},
                                                                            {"workload_start_idu", 0b1},
                                                                            {"workload_prm_sel", 0b0},
                                                                            {"workload_idu_auto_upd_0", 0b1}}},
                                                                  });

    if (_dryRunMode == VPU::DPUDryRunMode::STUB) {
        _log.trace("DPU dry run mode = 'stub', updating variant descriptor");
        fillStubCfg(initValues);
    } else {
        fillDPUConfigs(origOp.getRegion(), initValues);
    }
    fillBarrierCfg(origOp, initValues);
    fillProfilingCfg(origOp, initValues);

    auto taskListCfgOp = to_small_vector(origOp.getRegion().getOps<VPUIPDPU::DPUGroupOp>());
    if (!taskListCfgOp.empty()) {
        VPUX_THROW_UNLESS(taskListCfgOp.size() == 1, "Only one VPUIPDPU::DPUGroupOp should exist");
        auto tileSelectMask =
                1 << (taskListCfgOp[0].getInvariantIdx().getTileIdx() + NPUReg40XX::CMX_TILE_SELECT_OFFSET);
        VPURegMapped::updateRegMappedInitializationValues(
                initValues,
                {{"invariant_index_", {{"invariant_index_", taskListCfgOp[0].getInvariantIdx().getValue()}}},
                 {"invariant_", {{"invariant_", static_cast<uint64_t>(tileSelectMask)}}},
                 {"var_cfg", {{"invar_lptr_force", taskListCfgOp[0].getIsFirstVariant()}}},
                 {"dpu_cfg", {{"workload_odu_auto_upd", taskListCfgOp[0].getIsLastVariant()}}}});
    }

    if (origOp.getNextLinkAttr()) {
        VPURegMapped::updateRegMappedInitializationValues(initValues, {{"var_cfg", {{"next_sram_job_valid", 1}}}});
    }

    auto regDPUVariantAttr =
            VPURegMapped::getRegMappedAttributeWithValues<NPUReg40XX::RegMapped_DpuVariantRegisterType>(rewriter,
                                                                                                        initValues);

    rewriter.create<NPUReg40XX::DPUVariantOp>(origOp->getLoc(), origOp.getSymNameAttr(), origOp.getNextLinkAttr(),
                                              origOp.getTaskIndexAttr(), regDPUVariantAttr,
                                              origOp.getTaskLocationAttr(), origOp.getInvariantTaskLocationAttr(),
                                              origOp.getWeightsAttr(), origOp.getWeightTableAttr(),
                                              origOp.getNceTaskTypeAttr(), origOp.getWorkloadIdAttr());

    rewriter.eraseOp(origOp);

    return mlir::success();
}

void DPUVariantRewriter::fillDPUConfigs(mlir::Region& DPURegion,
                                        std::map<std::string, std::map<std::string, RegFieldValue>>& initValues) const {
    for (const auto& DPUOp : DPURegion.getOps()) {
        if (auto lowerToRegIfc = mlir::dyn_cast_or_null<VPUIPDPU::LowerToNPURegInterface>(&DPUOp)) {
            lowerToRegIfc.lowerToRegisters(initValues);
        }
    }
}

void DPUVariantRewriter::fillBarrierCfg(VPUIPDPU::DPUVariantOp origOp,
                                        std::map<std::string, std::map<std::string, RegFieldValue>>& initValues) const {
    VPUIPDPU::arch40xx::lowerToRegBarrierCfgOpWithDPUVariantParent(origOp, initValues);
}

void DPUVariantRewriter::fillProfilingCfg(
        VPUIPDPU::DPUVariantOp origOp,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
    if (!origOp.getWorkloadId().has_value()) {
        return;
    }
    uint32_t workloadId = origOp.getWorkloadId().value();
    VPURegMapped::updateRegMappedInitializationValues(
            initValues,
            {{"hwp_wload_id", {{"hwp_wload_id", workloadId}}},
             {"offset_addr",
              {{"odu_stat_en", 1}, {"idu_stat_en", 1}, {"idu_stat_clr_mode", 0}, {"odu_stat_clr_mode", 0}}}});
}

void DPUVariantRewriter::fillStubCfg(
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
    VPURegMapped::updateRegMappedInitializationValues(
            initValues, {
                                {"workload_size0", {{"workload_size_x", 0x1}, {"workload_size_y", 0x1}}},
                                {"workload_size1",
                                 {{"workload_size_z", 0x10},
                                  {"pad_count_up", 0x0},
                                  {"pad_count_left", 0x0},
                                  {"pad_count_down", 0x0},
                                  {"pad_count_right", 0x0}}},
                                {"workload_start0", {{"workload_start_x", 0x0}, {"workload_start_y", 0x0}}},
                                {"workload_start1", {{"workload_start_z", 0x0}}},
                                {"weight_size", {{"weight_size", 0x10}}},
                                {"weight_num", {{"weight_num", 0x10}}},
                                {"te_beg0", {{"te_beg_y", 0x0}, {"te_beg_z", 0x0}}},
                                {"te_beg1", {{"te_beg_x", 0x0}}},
                                {"te_end0", {{"te_end_y", 0x0}, {"te_end_z", 0xF}}},
                                {"te_end1", {{"te_end_x", 0x0}}},
                        });
}

}  // namespace vpuipdpu2npureg40xx
}  // namespace vpux
