//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUIPDPU2NPUReg40XX/dpu_invariant_rewriter.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/lower_to_registers.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"

using namespace vpux;
using namespace vpux::VPURegMapped;

namespace vpux {
namespace vpuipdpu2npureg40xx {

DPUInvariantRewriter::DPUInvariantRewriter(mlir::MLIRContext* ctx, Logger log, VPU::DPUDryRunMode dryRunMode)
        : mlir::OpRewritePattern<VPUIPDPU::DPUInvariantOp>(ctx), _log(log), _dryRunMode(dryRunMode) {
    setDebugName("DPUInvariant_VPUASM2NPUReg40XXRewriter");
}

mlir::LogicalResult DPUInvariantRewriter::matchAndRewrite(VPUIPDPU::DPUInvariantOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto initValues = NPUReg40XX::RegMapped_DpuInvariantRegisterType::getResetInitilizationValues();

    if (_dryRunMode == VPU::DPUDryRunMode::STUB) {
        _log.trace("DPU dry run mode = 'stub', updating invariant descriptor");
        fillStubCfg(initValues);
    } else {
        fillIDUCfg(origOp.getRegion(), initValues);
        fillMPECfg(origOp.getRegion(), initValues);
        fillPPECfg(origOp.getRegion(), initValues);
        fillODUCfg(origOp.getRegion(), initValues);
    }
    fillBarrierCfg(origOp, initValues);
    fillProfilingCfg(origOp, initValues);

    auto taskListCfgOp = to_small_vector(origOp.getRegion().getOps<VPUIPDPU::DPUGroupOp>());
    if (taskListCfgOp.size() == 1) {
        VPURegMapped::updateRegMappedInitializationValues(
                initValues, {{"variant_count_", {{"variant_count_", taskListCfgOp[0].getVariantCount()}}}});
    }

    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                      {{"nvar_tag", {{"nvar_tag", origOp.getIndex() + 1}}}});

    auto regDPUInvariantAttr =
            VPURegMapped::getRegMappedAttributeWithValues<NPUReg40XX::RegMapped_DpuInvariantRegisterType>(rewriter,
                                                                                                          initValues);

    rewriter.create<NPUReg40XX::DPUInvariantOp>(
            origOp->getLoc(), origOp.getSymNameAttr(), origOp.getTaskIndexAttr(), regDPUInvariantAttr,
            origOp.getTaskLocationAttr(), origOp.getInputAttr(), origOp.getInputSparsityMapAttr(),
            origOp.getInputStorageElementTableAttr(), origOp.getWeightsAttr(), origOp.getWeightsSparsityMapAttr(),
            origOp.getWeightTableAttr(), origOp.getSprLookupTableAttr(), origOp.getOutputAttr(),
            origOp.getOutputSparsityMapAttr(), origOp.getProfilingDataAttr(), origOp.getNceTaskTypeAttr(),
            origOp.getIsContinuedAttr());

    rewriter.eraseOp(origOp);

    return mlir::success();
}

void DPUInvariantRewriter::fillIDUCfg(
        mlir::Region& DPURegion,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
    auto IDUCfgOps = DPURegion.getOps<VPUIPDPU::IDUCfgOp>();
    if (!IDUCfgOps.empty()) {
        auto IDUCfgOp = *IDUCfgOps.begin();

        for (auto& IDUOp : IDUCfgOp.getRegion().getOps()) {
            if (auto lowerToRegIfc = mlir::dyn_cast_or_null<VPUIPDPU::LowerToNPURegInterface>(&IDUOp)) {
                lowerToRegIfc.lowerToRegisters(initValues);
            } else {
                VPUX_THROW("Missing interface to lower from VPUIPDPU to registers for IDU operation: {0}", IDUOp);
            }
        }
    }
}

void DPUInvariantRewriter::fillMPECfg(
        mlir::Region& DPURegion,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
    auto MPECfgOps = DPURegion.getOps<VPUIPDPU::MPECfgOp>();
    if (!MPECfgOps.empty()) {
        auto MPECfgOp = *MPECfgOps.begin();

        for (auto& MPEOp : MPECfgOp.getRegion().getOps()) {
            if (auto lowerToRegIfc = mlir::dyn_cast_or_null<VPUIPDPU::LowerToNPURegInterface>(&MPEOp)) {
                lowerToRegIfc.lowerToRegisters(initValues);
            } else {
                VPUX_THROW("Missing interface to lower from VPUIPDPU to registers for MPE operation: {0}", MPEOp);
            }
        }
    }
}

void DPUInvariantRewriter::fillPPECfg(
        mlir::Region& DPURegion,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
    auto PPECgfOps = DPURegion.getOps<VPUIPDPU::PPECfgOp>();
    if (!PPECgfOps.empty()) {
        auto PPECgfOp = *PPECgfOps.begin();

        for (auto& PPEOp : PPECgfOp.getRegion().getOps()) {
            if (auto lowerToRegIfc = mlir::dyn_cast_or_null<VPUIPDPU::LowerToNPURegInterface>(&PPEOp)) {
                lowerToRegIfc.lowerToRegisters(initValues);
            } else {
                VPUX_THROW("Missing interface to lower from VPUIPDPU to registers for PPE operation: {0}", PPEOp);
            }
        }
    }
}

void DPUInvariantRewriter::fillODUCfg(
        mlir::Region& DPURegion,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
    auto ODUCfgOps = DPURegion.getOps<VPUIPDPU::ODUCfgOp>();
    if (!ODUCfgOps.empty()) {
        auto ODUCfgOp = *ODUCfgOps.begin();

        // TODO: E#80766 select optimal write combine mode and serialize based on VPUIPDU instruction
        VPURegMapped::updateRegMappedInitializationValues(initValues, {{"odu_cfg", {{"wcb_bypass", 0}}}});

        // Statically set bits that should not be part of functional defaults

        // Not used by HW. Setting to 1 to be coeherent with GFile.
        VPURegMapped::updateRegMappedInitializationValues(initValues, {{"z_config", {{"addr_format_sel", 1}}}});
        // TODO: E#81883 need to figure this out why it's always set to 1?
        VPURegMapped::updateRegMappedInitializationValues(initValues, {{"kernel_pad_cfg", {{"rst_ctxt", 1}}}});

        // TODO: E#82814 should it be a  defailt value? this is hardcoded and directly copied from POC runtime...
        VPURegMapped::updateRegMappedInitializationValues(initValues, {{"base_offset_a", {{"base_offset_a", 0x200}}}});

        for (auto& ODUOp : ODUCfgOp.getRegion().getOps()) {
            if (auto lowerToRegIfc = mlir::dyn_cast_or_null<VPUIPDPU::LowerToNPURegInterface>(&ODUOp)) {
                lowerToRegIfc.lowerToRegisters(initValues);
            } else {
                VPUX_THROW("Missing interface to lower from VPUIPDPU to registers for ODU operation: {0}", ODUOp);
            }
        }
    }
}

void DPUInvariantRewriter::fillBarrierCfg(
        VPUIPDPU::DPUInvariantOp origOp,
        std::map<std::string, std::map<std::string, RegFieldValue>>& initValues) const {
    VPUIPDPU::arch40xx::lowerToRegBarrierCfgOpWithDPUInvariantParent(origOp, initValues);
}

void DPUInvariantRewriter::fillProfilingCfg(
        VPUIPDPU::DPUInvariantOp origOp,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
    if (!origOp.getProfilingData().has_value()) {
        return;
    }
    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                      {{"hwp_ctrl", {{"hwp_en", 1}, {"hwp_stat_mode", 3}}}});
}

void DPUInvariantRewriter::fillStubCfg(
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
    VPURegMapped::updateRegMappedInitializationValues(
            initValues, {
                                {"tensor_size0", {{"tensor_size_x", 0x1}, {"tensor_size_y", 0x1}}},
                                {"tensor_size1", {{"tensor_size_z", 0x10}}},
                                {"tensor_mode", {{"workload_operation", 0x0}, {"zm_input", 0x1}}},
                                {"kernel_pad_cfg", {{"kernel_y", 0x1}, {"kernel_x", 0x1}}},
                                {"elops_wload", {{"elop_wload", 0x1}, {"elop_wload_type", 0x1}}},
                                {"te_dim0", {{"te_dim_y", 0x0}, {"te_dim_z", 0xF}}},
                                {"te_dim1", {{"te_dim_x", 0x0}}},
                                {"odu_cfg", {{"nthw", 0x1}}},
                        });
}

}  // namespace vpuipdpu2npureg40xx
}  // namespace vpux
