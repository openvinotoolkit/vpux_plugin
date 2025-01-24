//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUIPDPU2NPUReg40XX/dpu_variant_rewriter.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/utils.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
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

template <typename halo_regionA, typename halo_regionB, typename halo_regionC, typename halo_regionD>
void fillValuesForHaloRegion(VPUIPDPU::ODUHaloRegionOp opHaloReg,
                             vpux::NPUReg40XX::Descriptors::DpuVariantRegister& descriptor) {
    uint64_t lsbWidthValue(0), msbWidthValue(0);
    computeLsbAndMsbFromTargetWidth(opHaloReg.getTargetWidth(), msbWidthValue, lsbWidthValue);

    descriptor.write<halo_regionA, Fields::sp_adr_offset>(opHaloReg.getSparsityOffset().value_or(0));
    descriptor.write<halo_regionA, Fields::tile_select>(static_cast<uint64_t>(opHaloReg.getCastToTile()));
    descriptor.write<halo_regionA, Fields::enable>(1);
    descriptor.write<halo_regionB, Fields::ac_adr_offset>(opHaloReg.getActivationsOffset());
    descriptor.write<halo_regionB, Fields::target_width_lsb>(lsbWidthValue);
    descriptor.write<halo_regionC, Fields::begin_x>(opHaloReg.getBeginCoordX());
    descriptor.write<halo_regionC, Fields::begin_y>(opHaloReg.getBeginCoordY());
    descriptor.write<halo_regionC, Fields::target_width_msb>(msbWidthValue);
    descriptor.write<halo_regionD, Fields::end_x>(opHaloReg.getEndCoordX());
    descriptor.write<halo_regionD, Fields::end_y>(opHaloReg.getEndCoordY());
}

void fillValuesForHaloRegion(uint8_t haloRegionIdx, VPUIPDPU::ODUHaloRegionOp opHaloReg,
                             vpux::NPUReg40XX::Descriptors::DpuVariantRegister& descriptor) {
    if (haloRegionIdx == 0) {
        fillValuesForHaloRegion<Registers::halo_region0A, Registers::halo_region0B, Registers::halo_region0C,
                                Registers::halo_region0D>(opHaloReg, descriptor);
    } else if (haloRegionIdx == 1) {
        fillValuesForHaloRegion<Registers::halo_region1A, Registers::halo_region1B, Registers::halo_region1C,
                                Registers::halo_region1D>(opHaloReg, descriptor);
    } else if (haloRegionIdx == 2) {
        fillValuesForHaloRegion<Registers::halo_region2A, Registers::halo_region2B, Registers::halo_region2C,
                                Registers::halo_region2D>(opHaloReg, descriptor);
    } else if (haloRegionIdx == 3) {
        fillValuesForHaloRegion<Registers::halo_region3A, Registers::halo_region3B, Registers::halo_region3C,
                                Registers::halo_region3D>(opHaloReg, descriptor);
    } else if (haloRegionIdx == 4) {
        fillValuesForHaloRegion<Registers::halo_region4A, Registers::halo_region4B, Registers::halo_region4C,
                                Registers::halo_region4D>(opHaloReg, descriptor);
    } else if (haloRegionIdx == 5) {
        fillValuesForHaloRegion<Registers::halo_region5A, Registers::halo_region5B, Registers::halo_region5C,
                                Registers::halo_region5D>(opHaloReg, descriptor);
    }
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

    // index incremented by one by runtime logic. Something to do with preemption
    // This value can be change if needed in the future. For now we use the index+1 just because it is
    // convinient for when we want to preempt when running on Simics (E71635)
    const uint64_t maxTaskId = (1ull << NPUReg40XX::RegField_var_tagType::getRegFieldWidth()) - 1;
    auto taskIdx = checked_cast_reg<NPUReg40XX::RegField_var_tagType>(
            static_cast<uint64_t>(origOp.getTaskIndex().getValue() % maxTaskId + 1));

    vpux::NPUReg40XX::Descriptors::DpuVariantRegister descriptor;
    descriptor.write<Fields::var_tag>(taskIdx);
    descriptor.write<Fields::workload_start_odu>(1);
    descriptor.write<Fields::workload_start_idu>(1);
    descriptor.write<Fields::workload_prm_sel>(0);
    descriptor.write<Fields::workload_idu_auto_upd_0>(1);
    //  Note: wt_swizzle_sel needs to be on by default to match NNRT GF behaviour
    descriptor.write<Fields::wt_swizzle_sel>(1);

    if (_dryRunMode == VPU::DPUDryRunMode::STUB) {
        _log.trace("DPU dry run mode = 'stub', updating variant descriptor");
        fillStubCfg(descriptor);
    } else {
        fillIDUCfg(origOp.getRegion(), descriptor);
        fillODUCfg(origOp.getRegion(), descriptor);
    }
    fillBarrierCfg(origOp, descriptor);
    fillProfilingCfg(origOp, descriptor);

    auto taskListCfgOp = to_small_vector(origOp.getRegion().getOps<VPUIPDPU::DPUGroupOp>());
    if (!taskListCfgOp.empty()) {
        VPUX_THROW_UNLESS(taskListCfgOp.size() == 1, "Only one VPUIPDPU::DPUGroupOp should exist");
        auto tileSelectMask = VPUMI40XX::generateTileMask({taskListCfgOp[0].getInvariantIdx().getTileIdx()});

        descriptor.write<Fields::invariant_index_>(taskListCfgOp[0].getInvariantIdx().getValue());
        descriptor.write<Fields::invariant_>(static_cast<uint64_t>(tileSelectMask));
        auto forceInvReadOp = to_small_vector(origOp.getRegion().getOps<VPUIPDPU::ForceInvReadOp>());
        descriptor.write<Fields::invar_lptr_force>(taskListCfgOp[0].getIsFirstVariant() || !forceInvReadOp.empty());
        descriptor.write<Fields::workload_odu_auto_upd>(taskListCfgOp[0].getIsLastVariant());
    }

    if (origOp.getNextLinkAttr()) {
        descriptor.write<Fields::next_sram_job_valid>(1);
    }

    auto regDPUVariantAttr = DpuVariantRegisterAttr::get(rewriter.getContext(), std::move(descriptor));

    rewriter.create<NPUReg40XX::DPUVariantOp>(origOp->getLoc(), origOp.getSymNameAttr(), origOp.getNextLinkAttr(),
                                              origOp.getTaskIndexAttr(), regDPUVariantAttr,
                                              origOp.getTaskLocationAttr(), origOp.getInvariantTaskLocationAttr(),
                                              origOp.getWeightsAttr(), origOp.getWeightTableAttr(),
                                              origOp.getNceTaskTypeAttr(), origOp.getWorkloadIdAttr());

    rewriter.eraseOp(origOp);

    return mlir::success();
}

void DPUVariantRewriter::fillIDUCfg(mlir::Region& DPURegion,
                                    vpux::NPUReg40XX::Descriptors::DpuVariantRegister& descriptor) const {
    for (const auto& DPUOp : DPURegion.getOps()) {
        // IDU ops
        if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUActSwizzleOp>(&DPUOp)) {
            descriptor.write<Fields::swizzle_key_offset>(static_cast<uint64_t>(op.getSwizzleKey()));
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUWeightSwizzleOp>(&DPUOp)) {
            descriptor.write<Fields::wt_swizzle_key>(static_cast<uint64_t>(op.getWtSwizzleKey()));
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUNthwNtkOp>(&DPUOp)) {
            descriptor.write<Fields::nthw_ntk>(static_cast<uint64_t>(op.getNthwNtk()));
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUSEDenseOp>(&DPUOp)) {
            descriptor.write<Fields::dense_se>(1);
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUConvContinueOp>(&DPUOp)) {
            descriptor.write<Fields::conv_cond>(1);
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUBinaryConfigOp>(&DPUOp)) {
            descriptor.write<Fields::bin_cfg>(1);
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUWorkloadSetOp>(&DPUOp)) {
            descriptor.write<Fields::workload_start_x>(op.getStartX());
            descriptor.write<Fields::workload_start_y>(op.getStartY());
            descriptor.write<Fields::workload_start_z>(op.getStartZ());
            descriptor.write<Fields::workload_size_x>(op.getSizeX());
            descriptor.write<Fields::workload_size_y>(op.getSizeY());
            descriptor.write<Fields::workload_size_z>(op.getSizeZ());
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUPaddingOp>(&DPUOp)) {
            auto padUp = op.getPadCount().getTop().getInt();
            auto padLeft = op.getPadCount().getLeft().getInt();
            auto padDown = op.getPadCount().getBottom().getInt();
            auto padRight = op.getPadCount().getRight().getInt();

            descriptor.write<Fields::pad_count_up>(padUp);
            descriptor.write<Fields::pad_count_left>(padLeft);
            descriptor.write<Fields::pad_count_down>(padDown);
            descriptor.write<Fields::pad_count_right>(padRight);
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUWeightSetOp>(&DPUOp)) {
            auto weightsNum = vpux::alignValUp(static_cast<std::uint64_t>(op.getWeightNum()),
                                               static_cast<std::uint64_t>(VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT));

            // weight_start register will be modified by relocation mechanism based on provided offset info
            descriptor.write<Fields::weight_size>(op.getWeightSize());
            descriptor.write<Fields::weight_num>(weightsNum);
            descriptor.write<Fields::weight_start>(op.getWeightStart());
        }
    }
}
void DPUVariantRewriter::fillODUCfg(mlir::Region& DPURegion,
                                    vpux::NPUReg40XX::Descriptors::DpuVariantRegister& descriptor) const {
    for (const auto& DPUOp : DPURegion.getOps()) {
        // ODU ops
        if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUOutSubtensorOp>(&DPUOp)) {
            descriptor.write<Fields::te_beg_y>(op.getBeginCoordY());
            descriptor.write<Fields::te_beg_z>(op.getBeginCoordZ());
            descriptor.write<Fields::te_beg_x>(op.getBeginCoordX());
            descriptor.write<Fields::te_end_y>(op.getEndCoordY());
            descriptor.write<Fields::te_end_z>(op.getEndCoordZ());
            descriptor.write<Fields::te_end_x>(op.getEndCoordX());
        } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUHaloCfgOp>(&DPUOp)) {
            uint8_t haloRegionIdx(0);
            for (const auto& haloRegionOp : op.getRegion().getOps()) {
                auto opHaloReg = mlir::dyn_cast_or_null<VPUIPDPU::ODUHaloRegionOp>(&haloRegionOp);
                if (opHaloReg == nullptr) {
                    VPUX_THROW("Found invalid child op under ODUHaloCfgOp: {0}", haloRegionOp);
                }

                fillValuesForHaloRegion(haloRegionIdx, opHaloReg, descriptor);
                haloRegionIdx++;
            }
        }
    }
}

void DPUVariantRewriter::fillBarrierCfg(VPUIPDPU::DPUVariantOp origOp,
                                        vpux::NPUReg40XX::Descriptors::DpuVariantRegister& descriptor) const {
    // TODO: E146560 - use barrierCfgOps only in variants that need to update barriers
    auto taskListCfgOps = to_small_vector(origOp.getRegion().getOps<VPUIPDPU::DPUGroupOp>());
    auto barrierCfgOps = to_small_vector(origOp.getRegion().getOps<VPUIPDPU::BarrierCfgOp>());

    if (barrierCfgOps.size() == 1 && taskListCfgOps.size() == 1) {
        auto taskListCfgOp = taskListCfgOps[0];
        auto variantCount = taskListCfgOp.getVariantCount();
        bool isFirstVariant = taskListCfgOp.getIsFirstVariant() || (variantCount == 1);
        bool isLastVariant = taskListCfgOp.getIsLastVariant() || (variantCount == 1);
        auto barrierCfgOp = barrierCfgOps[0];

        descriptor.write<Fields::pbarrier_hi>(0);
        descriptor.write<Fields::pbarrier_lo>(0);
        descriptor.write<Fields::cbarrier_hi>(0);
        descriptor.write<Fields::cbarrier_lo>(0);

        if (isFirstVariant) {
            auto consMaskLo = vpux::VPUMI40XX::computeMaskLo(barrierCfgOp.getWaitBarriers());
            auto consMaskHi = vpux::VPUMI40XX::computeMaskHi(barrierCfgOp.getWaitBarriers());
            descriptor.write<Fields::cbarrier_hi>(consMaskHi);
            descriptor.write<Fields::cbarrier_lo>(consMaskLo);
        }

        if (isLastVariant) {
            auto prodMaskLo = vpux::VPUMI40XX::computeMaskLo(barrierCfgOp.getUpdateBarriers());
            auto prodMaskHi = vpux::VPUMI40XX::computeMaskHi(barrierCfgOp.getUpdateBarriers());
            descriptor.write<Fields::pbarrier_hi>(prodMaskHi);
            descriptor.write<Fields::pbarrier_lo>(prodMaskLo);
        }
    } else {
        // just to explicitly show that we really intentionally only care about size == 1
        return;
    }

    return;
}

void DPUVariantRewriter::fillProfilingCfg(VPUIPDPU::DPUVariantOp origOp,
                                          vpux::NPUReg40XX::Descriptors::DpuVariantRegister& descriptor) const {
    if (!origOp.getWorkloadId().has_value()) {
        return;
    }
    uint32_t workloadId = origOp.getWorkloadId().value();

    descriptor.write<Fields::hwp_wload_id>(workloadId);
    descriptor.write<Fields::odu_stat_en>(1);
    descriptor.write<Fields::idu_stat_en>(1);
    descriptor.write<Fields::idu_stat_clr_mode>(0);
    descriptor.write<Fields::odu_stat_clr_mode>(0);
}

void DPUVariantRewriter::fillStubCfg(vpux::NPUReg40XX::Descriptors::DpuVariantRegister& descriptor) const {
    descriptor.write<Fields::workload_size_x>(0x1);
    descriptor.write<Fields::workload_size_y>(0x1);
    descriptor.write<Fields::workload_size_z>(0x10);
    descriptor.write<Fields::pad_count_up>(0x0);
    descriptor.write<Fields::pad_count_left>(0x0);
    descriptor.write<Fields::pad_count_down>(0x0);
    descriptor.write<Fields::pad_count_right>(0x0);
    descriptor.write<Fields::workload_start_x>(0x0);
    descriptor.write<Fields::workload_start_y>(0x0);
    descriptor.write<Fields::workload_start_z>(0x0);
    descriptor.write<Fields::weight_size>(0x10);
    descriptor.write<Fields::weight_num>(0x10);
    descriptor.write<Fields::te_beg_y>(0x0);
    descriptor.write<Fields::te_beg_z>(0x0);
    descriptor.write<Fields::te_beg_x>(0x0);
    descriptor.write<Fields::te_end_y>(0x0);
    descriptor.write<Fields::te_end_z>(0xF);
    descriptor.write<Fields::te_end_x>(0x0);
}

}  // namespace vpuipdpu2npureg40xx
}  // namespace vpux
