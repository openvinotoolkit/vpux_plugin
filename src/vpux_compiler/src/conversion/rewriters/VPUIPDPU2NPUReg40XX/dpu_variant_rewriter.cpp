//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUIPDPU2NPUReg40XX/dpu_variant_rewriter.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/compiler/utils/traits_utils.hpp"

using namespace vpux;
using namespace vpux::VPURegMapped;

namespace {

void computeLsbAndMsbFromTargetWidth(int64_t targetWidth, uint64_t& msbWidth, uint64_t& lsbWidth) {
    auto lsbBitWidth = NPUReg40XX::RegField_target_width_lsbType::getRegFieldWidth();
    auto msbBitWidth = NPUReg40XX::RegField_target_width_msbType::getRegFieldWidth();

    auto bitMask = (1 << (lsbBitWidth + msbBitWidth)) - 1;
    VPUX_THROW_WHEN(targetWidth & ~bitMask, "target_width value {0} is too big for {1} bits", targetWidth,
                    lsbBitWidth + msbBitWidth);

    auto bitMaskLsb = (1 << lsbBitWidth) - 1;
    lsbWidth = targetWidth & bitMaskLsb;

    auto bitMaskMsb = ((1 << msbBitWidth) - 1) << lsbBitWidth;
    msbWidth = (targetWidth & bitMaskMsb) >> lsbBitWidth;
}

}  // namespace

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
        fillIDUCfg(origOp.getRegion(), initValues);
        fillODUCfg(origOp.getRegion(), initValues);
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

void DPUVariantRewriter::fillIDUCfg(mlir::Region& DPURegion,
                                    std::map<std::string, std::map<std::string, uint64_t>>& initValues) const {
    for (const auto& DPUOp : DPURegion.getOps()) {
        // IDU ops
        if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUActSwizzleOp>(&DPUOp)) {
            auto swizzleKey = checked_cast_reg<NPUReg40XX::RegField_swizzle_key_offsetType>(op.getSwizzleKey());
            VPURegMapped::updateRegMappedInitializationValues(
                    initValues, {
                                        {"offset_addr", {{"swizzle_key_offset", swizzleKey}}},
                                });
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUWeightSwizzleOp>(&DPUOp)) {
            auto wtSwizzleKey = checked_cast_reg<NPUReg40XX::RegField_wt_swizzle_keyType>(op.getWtSwizzleKey());
            VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                              {{"offset_addr", {{"wt_swizzle_key", wtSwizzleKey}}}});
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUNthwNtkOp>(&DPUOp)) {
            auto nthwNtk = checked_cast_reg<NPUReg40XX::RegField_nthw_ntkType>(op.getNthwNtk());
            VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                              {
                                                                      {"offset_addr", {{"nthw_ntk", nthwNtk}}},
                                                              });
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUSEDenseOp>(&DPUOp)) {
            VPURegMapped::updateRegMappedInitializationValues(initValues, {
                                                                                  {"offset_addr", {{"dense_se", 1}}},
                                                                          });
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUConvContinueOp>(&DPUOp)) {
            VPURegMapped::updateRegMappedInitializationValues(initValues, {
                                                                                  {"offset_addr", {{"conv_cond", 1}}},
                                                                          });
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUBinaryConfigOp>(&DPUOp)) {
            VPURegMapped::updateRegMappedInitializationValues(initValues, {
                                                                                  {"offset_addr", {{"bin_cfg", 1}}},
                                                                          });
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUWorkloadSetOp>(&DPUOp)) {
            auto startX = checked_cast_reg<NPUReg40XX::RegField_workload_start_xType>(op.getStartX());
            auto startY = checked_cast_reg<NPUReg40XX::RegField_workload_start_yType>(op.getStartY());
            auto startZ = checked_cast_reg<NPUReg40XX::RegField_workload_start_zType>(op.getStartZ());
            auto sizeX = checked_cast_reg<NPUReg40XX::RegField_workload_size_xType>(op.getSizeX());
            auto sizeY = checked_cast_reg<NPUReg40XX::RegField_workload_size_yType>(op.getSizeY());
            auto sizeZ = checked_cast_reg<NPUReg40XX::RegField_workload_size_zType>(op.getSizeZ());

            VPURegMapped::updateRegMappedInitializationValues(
                    initValues, {{"workload_start0", {{"workload_start_x", startX}, {"workload_start_y", startY}}},
                                 {"workload_start1", {{"workload_start_z", startZ}}},
                                 {"workload_size0", {{"workload_size_x", sizeX}, {"workload_size_y", sizeY}}},
                                 {"workload_size1", {{"workload_size_z", sizeZ}}}});
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUPaddingOp>(&DPUOp)) {
            auto padUp = checked_cast_reg<NPUReg40XX::RegField_pad_count_upType>(op.getPadCount().getTop().getInt());
            auto padLeft =
                    checked_cast_reg<NPUReg40XX::RegField_pad_count_leftType>(op.getPadCount().getLeft().getInt());
            auto padDown =
                    checked_cast_reg<NPUReg40XX::RegField_pad_count_downType>(op.getPadCount().getBottom().getInt());
            auto padRight =
                    checked_cast_reg<NPUReg40XX::RegField_pad_count_rightType>(op.getPadCount().getRight().getInt());
            VPURegMapped::updateRegMappedInitializationValues(initValues, {
                                                                                  {"workload_size1",
                                                                                   {{"pad_count_up", padUp},
                                                                                    {"pad_count_left", padLeft},
                                                                                    {"pad_count_down", padDown},
                                                                                    {"pad_count_right", padRight}}},
                                                                          });
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUWeightSetOp>(&DPUOp)) {
            auto weightsStart = checked_cast_reg<NPUReg40XX::RegField_weight_startType>(op.getWeightStart());
            auto weightsNum = vpux::alignValUp(checked_cast_reg<NPUReg40XX::RegField_weight_numType>(op.getWeightNum()),
                                               static_cast<std::uint64_t>(VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT));
            auto weightsSize = checked_cast_reg<NPUReg40XX::RegField_weight_sizeType>(op.getWeightSize());

            // weight_start register will be modified by relocation mechanism based on provided offset info
            VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                              {{"weight_size", {{"weight_size", weightsSize}}},
                                                               {"weight_num", {{"weight_num", weightsNum}}},
                                                               {"weight_start", {{"weight_start", weightsStart}}}});
        }
    }
}
void DPUVariantRewriter::fillODUCfg(mlir::Region& DPURegion,
                                    std::map<std::string, std::map<std::string, uint64_t>>& initValues) const {
    uint8_t haloRegionIdx(0);
    for (const auto& DPUOp : DPURegion.getOps()) {
        // ODU ops
        if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUOutSubtensorOp>(&DPUOp)) {
            auto begX = checked_cast_reg<NPUReg40XX::RegField_te_beg_xType>(op.getBeginCoordX());
            auto begY = checked_cast_reg<NPUReg40XX::RegField_te_beg_yType>(op.getBeginCoordY());
            auto begZ = checked_cast_reg<NPUReg40XX::RegField_te_beg_zType>(op.getBeginCoordZ());
            auto endX = checked_cast_reg<NPUReg40XX::RegField_te_end_xType>(op.getEndCoordX());
            auto endY = checked_cast_reg<NPUReg40XX::RegField_te_end_yType>(op.getEndCoordY());
            auto endZ = checked_cast_reg<NPUReg40XX::RegField_te_end_zType>(op.getEndCoordZ());
            VPURegMapped::updateRegMappedInitializationValues(
                    initValues, {
                                        {"te_beg0", {{"te_beg_y", begY}, {"te_beg_z", begZ}}},
                                        {"te_beg1", {{"te_beg_x", begX}}},
                                        {"te_end0", {{"te_end_y", endY}, {"te_end_z", endZ}}},
                                        {"te_end1", {{"te_end_x", endX}}},
                                });
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUHaloRegionOp>(&DPUOp)) {
            auto begX = checked_cast_reg<NPUReg40XX::RegField_begin_xType>(op.getBeginCoordX());
            auto begY = checked_cast_reg<NPUReg40XX::RegField_begin_yType>(op.getBeginCoordY());
            auto endX = checked_cast_reg<NPUReg40XX::RegField_end_xType>(op.getEndCoordX());
            auto endY = checked_cast_reg<NPUReg40XX::RegField_end_xType>(op.getEndCoordY());
            auto actOffset = checked_cast_reg<NPUReg40XX::RegField_ac_adr_offsetType>(op.getActivationsOffset());

            uint64_t lsbWidthValue(0), msbWidthValue(0);
            computeLsbAndMsbFromTargetWidth(op.getTargetWidth(), msbWidthValue, lsbWidthValue);

            auto targetWidthLsb = checked_cast_reg<NPUReg40XX::RegField_target_width_lsbType>(lsbWidthValue);
            auto targetWidthMsb = checked_cast_reg<NPUReg40XX::RegField_target_width_msbType>(msbWidthValue);
            auto castToTile = checked_cast_reg<NPUReg40XX::RegField_tile_selectType>(op.getCastToTile());
            auto sparsityOffset =
                    checked_cast_reg<NPUReg40XX::RegField_sp_adr_offsetType>(op.getSparsityOffset().value_or(0));

            auto haloRegionA = std::string("halo_region" + std::to_string(haloRegionIdx) + "A");
            auto haloRegionB = std::string("halo_region" + std::to_string(haloRegionIdx) + "B");
            auto haloRegionC = std::string("halo_region" + std::to_string(haloRegionIdx) + "C");
            auto haloRegionD = std::string("halo_region" + std::to_string(haloRegionIdx) + "D");

            VPURegMapped::updateRegMappedInitializationValues(
                    initValues,
                    {
                            {haloRegionA,
                             {{"sp_adr_offset", sparsityOffset}, {"tile_select", castToTile}, {"enable", 1}}},
                            {haloRegionB, {{"ac_adr_offset", actOffset}, {"target_width_lsb", targetWidthLsb}}},
                            {haloRegionC, {{"begin_x", begX}, {"begin_y", begY}, {"target_width_msb", targetWidthMsb}}},
                            {haloRegionD, {{"end_x", endX}, {"end_y", endY}}},
                    });

            haloRegionIdx++;
        }
    }
}

void DPUVariantRewriter::fillBarrierCfg(VPUIPDPU::DPUVariantOp origOp,
                                        std::map<std::string, std::map<std::string, uint64_t>>& initValues) const {
    auto barrierCfgOps = to_small_vector(origOp.getRegion().getOps<VPUIPDPU::BarrierCfgOp>());
    if (barrierCfgOps.size() == 1) {
        auto barrierCfgOp = barrierCfgOps[0];

        uint64_t prodMaskLo = vpux::VPUMI40XX::computeMaskLo(barrierCfgOp.getUpdateBarriers());
        uint64_t prodMaskHi = vpux::VPUMI40XX::computeMaskHi(barrierCfgOp.getUpdateBarriers());
        uint64_t consMaskLo = vpux::VPUMI40XX::computeMaskLo(barrierCfgOp.getWaitBarriers());
        uint64_t consMaskHi = vpux::VPUMI40XX::computeMaskHi(barrierCfgOp.getWaitBarriers());

        VPURegMapped::updateRegMappedInitializationValues(initValues, {{"cbarrier_hi", {{"cbarrier_hi", consMaskHi}}},
                                                                       {"cbarrier_lo", {{"cbarrier_lo", consMaskLo}}},
                                                                       {"pbarrier_hi", {{"pbarrier_hi", prodMaskHi}}},
                                                                       {"pbarrier_lo", {{"pbarrier_lo", prodMaskLo}}}});
    } else {
        // just to explicitly show that we really intentionally only care about size == 1
        return;
    }

    return;
}

void DPUVariantRewriter::fillProfilingCfg(VPUIPDPU::DPUVariantOp origOp,
                                          std::map<std::string, std::map<std::string, uint64_t>>& initValues) const {
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

void DPUVariantRewriter::fillStubCfg(std::map<std::string, std::map<std::string, uint64_t>>& initValues) const {
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
