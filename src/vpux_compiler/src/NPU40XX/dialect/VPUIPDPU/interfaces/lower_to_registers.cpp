//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/lower_to_registers.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIPDPU/ops_interfaces.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"

namespace vpux::VPUIPDPU::arch40xx {

bool lowerToRegIDUWorkloadCfgOp(
        VPUIPDPU::IDUWorkloadCfgOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    bool successfullyLowered = true;
    auto workloadType = op.getWorkloadType();
    switch (workloadType) {
    case VPUIPDPU::IDUWorkloadType::MAXPOOL:
        VPURegMapped::updateRegMappedInitializationValues(
                initValues, {{"tensor_mode", {{"workload_operation", 0b10}, {"zm_input", 0b1}, {"dw_input", 0b1}}},
                             {"elops_wload", {{"pool_wt_rd_dis", 0b1}}},
                             {"kernel_pad_cfg", {{"dw_wt_sp_ins", 0b1}}}});
        break;
    case VPUIPDPU::IDUWorkloadType::AVEPOOL:
        VPURegMapped::updateRegMappedInitializationValues(
                initValues,
                {{"tensor_mode", {{"workload_operation", 0}, {"zm_input", 0b1}, {"dw_input", 0b1}}},  // CONV workload
                 {"elops_wload", {{"pool_wt_rd_dis", 0b1}}},
                 {"kernel_pad_cfg", {{"dw_wt_sp_ins", 0b1}, {"dynamic_bw_en", 0b1}}}});

        break;
    case VPUIPDPU::IDUWorkloadType::CONV:
        VPURegMapped::updateRegMappedInitializationValues(
                initValues, {{"tensor_mode", {{"workload_operation", 0}, {"zm_input", 0b1}}}});  // CONV workload
        VPURegMapped::updateRegMappedInitializationValues(initValues, {{"kernel_pad_cfg", {{"dynamic_bw_en", 1}}}});
        break;
    case VPUIPDPU::IDUWorkloadType::DWCONV:
        VPURegMapped::updateRegMappedInitializationValues(
                initValues,
                {{"tensor_mode", {{"workload_operation", 0}, {"zm_input", 0b1}, {"dw_input", 0b1}}},  // CONV workload
                 {"kernel_pad_cfg", {{"dw_wt_sp_ins", 0b1}, {"dynamic_bw_en", 0b1}}}});
        break;
    case VPUIPDPU::IDUWorkloadType::ELTWISE:
        VPURegMapped::updateRegMappedInitializationValues(
                initValues, {{"tensor_mode", {{"workload_operation", 0}, {"zm_input", 0b1}}},  // CONV workload
                             {"kernel_pad_cfg", {{"dynamic_bw_en", 0b1}}},
                             {"elops_wload", {{"elop_wload", 0b1}}}});
        break;
    default:
        successfullyLowered = false;
        break;
    }
    return successfullyLowered;
}

void lowerToRegPPEIntClampOp(
        VPUIPDPU::PPEIntClampOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto clampHigh = checked_cast_reg<NPUReg40XX::RegField_ppe_scale_hclampType>(op.getClampHigh());
    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                      {
                                                              {"ppe_scale_hclamp", {{"ppe_scale_hclamp", clampHigh}}},
                                                      });
    if (op.getClampLow().has_value()) {
        auto clampLow = checked_cast_reg<NPUReg40XX::RegField_ppe_scale_lclampType>(op.getClampLow().value());
        VPURegMapped::updateRegMappedInitializationValues(
                initValues, {
                                    {"ppe_scale_lclamp", {{"ppe_scale_lclamp", clampLow}}},
                            });
    }
}

void lowerToRegBarrierCfgOpWithDPUInvariantParent(
        VPUIPDPU::DPUInvariantOp& dpuInvariantOp,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto barrierCfgOps = to_small_vector(dpuInvariantOp.getRegion().getOps<VPUIPDPU::BarrierCfgOp>());
    if (barrierCfgOps.size() == 1) {
        auto barrierCfgOp = barrierCfgOps[0];

        uint64_t prodMaskLo = vpux::VPUMI40XX::computeMaskLo(barrierCfgOp.getUpdateBarriers());
        uint64_t prodMaskHi = vpux::VPUMI40XX::computeMaskHi(barrierCfgOp.getUpdateBarriers());
        uint64_t consMaskLo = vpux::VPUMI40XX::computeMaskLo(barrierCfgOp.getWaitBarriers());
        uint64_t consMaskHi = vpux::VPUMI40XX::computeMaskHi(barrierCfgOp.getWaitBarriers());

        uint64_t startAfter = barrierCfgOp.getStartAfter();
        uint64_t cleanAfter = barrierCfgOp.getCleanAfter();

        uint8_t barrierGroup = 0;
        uint8_t barrierMask = 0;
        std::tie(barrierGroup, barrierMask) = ELF::reduceWaitMaskTo8bit(consMaskLo);

        VPURegMapped::updateRegMappedInitializationValues(
                initValues, {{"barriers_group_mask_", {{"group_", barrierGroup}, {"mask_", barrierMask}}},
                             {"barriers_sched_", {{"start_after_", startAfter}, {"clean_after_", cleanAfter}}},
                             {"barriers_wait_mask_hi_", {{"barriers_wait_mask_hi_", consMaskHi}}},
                             {"barriers_wait_mask_lo_", {{"barriers_wait_mask_lo_", consMaskLo}}},
                             {"barriers_post_mask_hi_", {{"barriers_post_mask_hi_", prodMaskHi}}},
                             {"barriers_post_mask_lo_", {{"barriers_post_mask_lo_", prodMaskLo}}}});
    } else {
        // just to explicitly show that we really intentionally only care about size == 1
        return;
    }
}

void lowerToRegBarrierCfgOpWithDPUVariantParent(
        VPUIPDPU::DPUVariantOp& dpuVariantOp,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto barrierCfgOps = to_small_vector(dpuVariantOp.getRegion().getOps<VPUIPDPU::BarrierCfgOp>());
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
}

}  // namespace vpux::VPUIPDPU::arch40xx

// Implementations of the lowering interface that do not change between architectures:
// (interface declaration for ops is present in tblgen)

void vpux::VPUIPDPU::IDUSEDenseOp::lowerToRegisters(
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    VPURegMapped::updateRegMappedInitializationValues(initValues, {
                                                                          {"offset_addr", {{"dense_se", 1}}},
                                                                  });
}

void vpux::VPUIPDPU::MPEDenormalOperandsFTZOp::lowerToRegisters(
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    VPURegMapped::updateRegMappedInitializationValues(initValues, {{"mpe_cfg", {{"mpe_daz", 1}}}});
}

void vpux::VPUIPDPU::IDUConvContinueOp::lowerToRegisters(
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    VPURegMapped::updateRegMappedInitializationValues(initValues, {
                                                                          {"offset_addr", {{"conv_cond", 1}}},
                                                                  });
}

void vpux::VPUIPDPU::IDUBinaryConfigOp::lowerToRegisters(
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    VPURegMapped::updateRegMappedInitializationValues(initValues, {
                                                                          {"offset_addr", {{"bin_cfg", 1}}},
                                                                  });
}

// Implementations of the lowering interface that are specific to arch 40xx:

namespace {

using RegistersIDUStorageElementOp = struct RegistersIDUStorageElementOp {
    using TRegField_se_z_splitType = NPUReg40XX::RegField_se_z_splitType;
    using TRegField_npo2_se_sizeType = NPUReg40XX::RegField_npo2_se_sizeType;
    using TRegField_num_ses_in_z_dirType = NPUReg40XX::RegField_num_ses_in_z_dirType;
};

class LowerToRegIDUStorageElementOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegIDUStorageElementOpInterfaceModel,
                                                               VPUIPDPU::IDUStorageElementOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegIDUStorageElementOp<RegistersIDUStorageElementOp>(
                mlir::cast<VPUIPDPU::IDUStorageElementOp>(op), initValues);
    }
};

using RegistersIDUKernelOp = struct RegistersIDUKernelOp {
    using TRegField_kernel_xType = NPUReg40XX::RegField_kernel_xType;
    using TRegField_kernel_yType = NPUReg40XX::RegField_kernel_yType;
};

class LowerToRegIDUKernelOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegIDUKernelOpInterfaceModel,
                                                               VPUIPDPU::IDUKernelOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegIDUKernelOp<RegistersIDUKernelOp>(mlir::cast<VPUIPDPU::IDUKernelOp>(op),
                                                                        initValues);
    }
};

using RegistersIDUStrideOp = struct RegistersIDUStrideOp {
    using TRegField_stride_yType = NPUReg40XX::RegField_stride_yType;
    using TRegField_strideType = NPUReg40XX::RegField_strideType;
};

class LowerToRegIDUStrideOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegIDUStrideOpInterfaceModel,
                                                               VPUIPDPU::IDUStrideOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegIDUStrideOp<RegistersIDUStrideOp>(mlir::cast<VPUIPDPU::IDUStrideOp>(op),
                                                                        initValues);
    }
};

using TensorModeAcceptedRegisters = struct TensorModeAcceptedRegisters {
    using TRegField_amodeType = NPUReg40XX::RegField_amodeType;
    using TRegField_wmodeType = NPUReg40XX::RegField_wmodeType;
    using TRegField_dma_acc_info_compress_dtypeType = NPUReg40XX::RegField_dma_acc_info_compress_dtypeType;
    using TRegField_dma_acc_info_decompress_dtypeType = NPUReg40XX::RegField_dma_acc_info_decompress_dtypeType;
};

using RegistersIDUInActivationsOp = struct RegistersIDUInActivationsOp {
    using TRegField_tensor_size_yType = NPUReg40XX::RegField_tensor_size_yType;
    using TRegField_tensor_size_xType = NPUReg40XX::RegField_tensor_size_xType;
    using TRegField_tensor_size_zType = NPUReg40XX::RegField_tensor_size_zType;
    using TRegField_amodeType = NPUReg40XX::RegField_amodeType;
    using TRegField_act_denseType = NPUReg40XX::RegField_act_denseType;
    using TTensorModeAcceptedRegisters = TensorModeAcceptedRegisters;
};

class LowerToRegIDUInActivationsOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegIDUInActivationsOpInterfaceModel,
                                                               VPUIPDPU::IDUInActivationsOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegIDUInActivationsOp<RegistersIDUInActivationsOp>(
                mlir::cast<VPUIPDPU::IDUInActivationsOp>(op), initValues);
    }
};

using RegistersIDUInputLayerCfgOp = struct RegistersIDUInputLayerCfgOp {
    using TRegField_cm_sp_patternType = NPUReg40XX::RegField_cm_sp_patternType;
    using TRegField_layer1_cmp_enType = NPUReg40XX::RegField_layer1_cmp_enType;
};

class LowerToRegIDUInputLayerCfgOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegIDUInputLayerCfgOpInterfaceModel,
                                                               VPUIPDPU::IDUInputLayerCfgOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegIDUInputLayerCfgOp<RegistersIDUInputLayerCfgOp>(
                mlir::cast<VPUIPDPU::IDUInputLayerCfgOp>(op), initValues);
    }
};

using RegistersIDUWeightsOp = struct RegistersIDUWeightsOp {
    using TRegField_wmodeType = NPUReg40XX::RegField_wmodeType;
    using TRegField_wt_denseType = NPUReg40XX::RegField_wt_denseType;
    using TRegField_pool_wt_dataType = NPUReg40XX::RegField_pool_wt_dataType;
    using TTensorModeAcceptedRegisters = TensorModeAcceptedRegisters;
};

class LowerToRegIDUWeightsOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegIDUWeightsOpInterfaceModel,
                                                               VPUIPDPU::IDUWeightsOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegIDUWeightsOp<RegistersIDUWeightsOp>(mlir::cast<VPUIPDPU::IDUWeightsOp>(op),
                                                                          initValues);
    }
};

class LowerToRegIDUWorkloadCfgOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegIDUWorkloadCfgOpInterfaceModel,
                                                               VPUIPDPU::IDUWorkloadCfgOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        if (!VPUIPDPU::arch40xx::lowerToRegIDUWorkloadCfgOp(mlir::cast<VPUIPDPU::IDUWorkloadCfgOp>(op), initValues)) {
            VPUX_THROW("Failed to lower to registers the op IDUWorkloadCfg: invalid Wload Type");
        }
    }
};

class LowerToRegIDUDepthWiseCfgOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegIDUDepthWiseCfgOpInterfaceModel,
                                                               VPUIPDPU::IDUDepthWiseCfgOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegIDUDepthWiseCfgOp<NPUReg40XX::RegField_dw_opt_offsetType>(
                mlir::cast<VPUIPDPU::IDUDepthWiseCfgOp>(op), initValues);
    }
};

using RegistersIDUEltWiseCfgOp = struct RegistersIDUEltWiseCfgOp {
    using TRegField_elop_scale_aType = NPUReg40XX::RegField_elop_scale_aType;
    using TRegField_elop_scale_bType = NPUReg40XX::RegField_elop_scale_bType;
};

class LowerToRegIDUEltWiseCfgOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegIDUEltWiseCfgOpInterfaceModel,
                                                               VPUIPDPU::IDUEltWiseCfgOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegIDUEltWiseCfgOp<RegistersIDUEltWiseCfgOp>(
                mlir::cast<VPUIPDPU::IDUEltWiseCfgOp>(op), initValues);
    }
};

class LowerToRegIDUActSwizzleOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegIDUActSwizzleOpInterfaceModel,
                                                               VPUIPDPU::IDUActSwizzleOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegIDUActSwizzleOp<NPUReg40XX::RegField_swizzle_key_offsetType>(
                mlir::cast<VPUIPDPU::IDUActSwizzleOp>(op), initValues);
    }
};

class LowerToRegIDUWeightSwizzleOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegIDUWeightSwizzleOpInterfaceModel,
                                                               VPUIPDPU::IDUWeightSwizzleOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegIDUWeightSwizzleOp<NPUReg40XX::RegField_wt_swizzle_keyType>(
                mlir::cast<VPUIPDPU::IDUWeightSwizzleOp>(op), initValues);
    }
};

class LowerToRegIDUNthwNtkOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegIDUNthwNtkOpInterfaceModel,
                                                               VPUIPDPU::IDUNthwNtkOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegIDUNthwNtkOp<NPUReg40XX::RegField_nthw_ntkType>(
                mlir::cast<VPUIPDPU::IDUNthwNtkOp>(op), initValues);
    }
};

using RegistersIDUWorkloadSetOp = struct RegistersIDUWorkloadSetOp {
    using TRegField_workload_start_xType = NPUReg40XX::RegField_workload_start_xType;
    using TRegField_workload_start_yType = NPUReg40XX::RegField_workload_start_yType;
    using TRegField_workload_start_zType = NPUReg40XX::RegField_workload_start_zType;
    using TRegField_workload_size_xType = NPUReg40XX::RegField_workload_size_xType;
    using TRegField_workload_size_yType = NPUReg40XX::RegField_workload_size_yType;
    using TRegField_workload_size_zType = NPUReg40XX::RegField_workload_size_zType;
};

class LowerToRegIDUWorkloadSetOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegIDUWorkloadSetOpInterfaceModel,
                                                               VPUIPDPU::IDUWorkloadSetOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegIDUWorkloadSetOp<RegistersIDUWorkloadSetOp>(
                mlir::cast<VPUIPDPU::IDUWorkloadSetOp>(op), initValues);
    }
};

using RegistersIDUPaddingOp = struct RegistersIDUPaddingOp {
    using TRegField_pad_count_upType = NPUReg40XX::RegField_pad_count_upType;
    using TRegField_pad_count_leftType = NPUReg40XX::RegField_pad_count_leftType;
    using TRegField_pad_count_downType = NPUReg40XX::RegField_pad_count_downType;
    using TRegField_pad_count_rightType = NPUReg40XX::RegField_pad_count_rightType;
};

class LowerToRegIDUPaddingOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegIDUPaddingOpInterfaceModel,
                                                               VPUIPDPU::IDUPaddingOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegIDUPaddingOp<RegistersIDUPaddingOp>(mlir::cast<VPUIPDPU::IDUPaddingOp>(op),
                                                                          initValues);
    }
};

using RegistersIDUWeightSetOp = struct RegistersIDUWeightSetOp {
    using TRegField_weight_startType = NPUReg40XX::RegField_weight_startType;
    using TRegField_weight_numType = NPUReg40XX::RegField_weight_numType;
    using TRegField_weight_sizeType = NPUReg40XX::RegField_weight_sizeType;
};

class LowerToRegIDUWeightSetOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegIDUWeightSetOpInterfaceModel,
                                                               VPUIPDPU::IDUWeightSetOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegIDUWeightSetOp<RegistersIDUWeightSetOp>(mlir::cast<VPUIPDPU::IDUWeightSetOp>(op),
                                                                              initValues);
    }
};

class LowerToRegMpeActivationBiasOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegMpeActivationBiasOpInterfaceModel,
                                                               VPUIPDPU::MPEActivationBiasOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegMPEActivationBiasOp<NPUReg40XX::RegField_mpe_actbiasType>(
                mlir::cast<VPUIPDPU::MPEActivationBiasOp>(op), initValues);
    }
};

class LowerToRegMpeWeightsBiasOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegMpeWeightsBiasOpInterfaceModel,
                                                               VPUIPDPU::MPEWeightsBiasOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegMPEWeightsBiasOp<NPUReg40XX::RegField_mpe_wtbiasType>(
                mlir::cast<VPUIPDPU::MPEWeightsBiasOp>(op), initValues);
    }
};

class LowerToRegPPEFpBiasAddOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegPPEFpBiasAddOpInterfaceModel,
                                                               VPUIPDPU::PPEFpBiasAddOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegPPEFpBiasAddOp<NPUReg40XX::RegField_ppe_fp_biasType>(
                mlir::cast<VPUIPDPU::PPEFpBiasAddOp>(op), initValues);
    }
};

using RegistersPPEFpScalePreluMultOp = struct RegistersPPEFpScalePreluMultOp {
    using TRegField_ppe_fp_scaleType = NPUReg40XX::RegField_ppe_fp_scaleType;
    using TRegField_ppe_fp_preluType = NPUReg40XX::RegField_ppe_fp_preluType;
};

class LowerToRegPPEFpScalePreluMultOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegPPEFpScalePreluMultOpInterfaceModel,
                                                               VPUIPDPU::PPEFpScalePreluMultOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegPPEFpScalePreluMultOp<RegistersPPEFpScalePreluMultOp>(
                mlir::cast<VPUIPDPU::PPEFpScalePreluMultOp>(op), initValues);
    }
};

class LowerToRegPPEFpAddMultBypassOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegPPEFpAddMultBypassOpInterfaceModel,
                                                               VPUIPDPU::PPEFpAddMultBypassOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegPPEFpAddMultBypassOp<NPUReg40XX::RegField_ppe_fp_bypassType>(
                mlir::cast<VPUIPDPU::PPEFpAddMultBypassOp>(op), initValues);
    }
};

using RegistersPPEFpConvertOp = struct RegistersPPEFpConvertOp {
    using TRegField_ppe_fp_convertType = NPUReg40XX::RegField_ppe_fp_convertType;
    using TRegField_ppe_fp16_clampType = NPUReg40XX::RegField_ppe_fp16_clampType;
    using TRegField_ppe_fp16_ftzType = NPUReg40XX::RegField_ppe_fp16_ftzType;
    using TRegField_ppe_bf16_roundType = NPUReg40XX::RegField_ppe_bf16_roundType;
};

class LowerToRegPPEFpConvertOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegPPEFpConvertOpInterfaceModel,
                                                               VPUIPDPU::PPEFpConvertOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegPPEFpScalePreluMultOp<RegistersPPEFpConvertOp>(
                mlir::cast<VPUIPDPU::PPEFpConvertOp>(op), initValues);
    }
};

class LowerToRegPPEIntBiasAddOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegPPEIntBiasAddOpInterfaceModel,
                                                               VPUIPDPU::PPEIntBiasAddOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegPPEIntBiasAddOp<NPUReg40XX::RegField_ppe_biasType>(
                mlir::cast<VPUIPDPU::PPEIntBiasAddOp>(op), initValues);
    }
};

class LowerToRegPPEIntScaleMultOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegPPEIntScaleMultOpInterfaceModel,
                                                               VPUIPDPU::PPEIntScaleMultOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegPPEIntScaleMultOp<NPUReg40XX::RegField_ppe_scale_multType>(
                mlir::cast<VPUIPDPU::PPEIntScaleMultOp>(op), initValues);
    }
};

class LowerToRegPPEIntPreluMultOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegPPEIntPreluMultOpInterfaceModel,
                                                               VPUIPDPU::PPEIntPreluMultOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegPPEIntPreluMultOp<NPUReg40XX::RegField_ppe_prelu_multType>(
                mlir::cast<VPUIPDPU::PPEIntPreluMultOp>(op), initValues);
    }
};

class LowerToRegPPEIntScaleShiftOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegPPEIntScaleShiftOpInterfaceModel,
                                                               VPUIPDPU::PPEIntScaleShiftOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegPPEIntScaleShiftOp<NPUReg40XX::RegField_ppe_scale_shiftType>(
                mlir::cast<VPUIPDPU::PPEIntScaleShiftOp>(op), initValues);
    }
};

class LowerToRegPPEIntPreluShiftOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegPPEIntPreluShiftOpInterfaceModel,
                                                               VPUIPDPU::PPEIntPreluShiftOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegPPEIntPreluShiftOp<NPUReg40XX::RegField_ppe_prelu_shiftType>(
                mlir::cast<VPUIPDPU::PPEIntPreluShiftOp>(op), initValues);
    }
};

class LowerToRegPPEIntRoundOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegPPEIntRoundOpInterfaceModel,
                                                               VPUIPDPU::PPEIntRoundOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegPPEIntRoundOp<NPUReg40XX::RegField_ppe_scale_roundType>(
                mlir::cast<VPUIPDPU::PPEIntRoundOp>(op), initValues);
    }
};

class LowerToRegPPEIntZeroPointOffsetOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegPPEIntZeroPointOffsetOpInterfaceModel,
                                                               VPUIPDPU::PPEIntZeroPointOffsetOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegPPEIntZeroPointOffsetOp<NPUReg40XX::RegField_ppe_g8_bias_cType>(
                mlir::cast<VPUIPDPU::PPEIntZeroPointOffsetOp>(op), initValues);
    }
};

class LowerToRegPPEIntClampOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegPPEIntClampOpInterfaceModel,
                                                               VPUIPDPU::PPEIntClampOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegPPEIntClampOp(mlir::cast<VPUIPDPU::PPEIntClampOp>(op), initValues);
    }
};

class LowerToRegPPEIntConvertOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegPPEIntConvertOpInterfaceModel,
                                                               VPUIPDPU::PPEIntConvertOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegPPEIntConvertOp<NPUReg40XX::RegField_ppe_i32_convertType>(
                mlir::cast<VPUIPDPU::PPEIntConvertOp>(op), initValues);
    }
};

using RegistersODUOutTensorSizeOp = struct RegistersODUOutTensorSizeOp {
    using TRegField_te_dim_xType = NPUReg40XX::RegField_te_dim_xType;
    using TRegField_te_dim_yType = NPUReg40XX::RegField_te_dim_yType;
    using TRegField_te_dim_zType = NPUReg40XX::RegField_te_dim_zType;
};

class LowerToRegODUOutTensorSizeOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegODUOutTensorSizeOpInterfaceModel,
                                                               VPUIPDPU::ODUOutTensorSizeOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegODUOutTensorSizeOp<RegistersODUOutTensorSizeOp>(
                mlir::cast<VPUIPDPU::ODUOutTensorSizeOp>(op), initValues);
    }
};

class LowerToRegODUDataReuseOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegODUDataReuseOpInterfaceModel,
                                                               VPUIPDPU::ODUDataReuseOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegODUDataReuseOp<NPUReg40XX::RegField_nthwType>(
                mlir::cast<VPUIPDPU::ODUDataReuseOp>(op), initValues);
    }
};

class LowerToRegODUPermuteDataOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegODUPermuteDataOpInterfaceModel,
                                                               VPUIPDPU::ODUPermuteDataOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegODUPermuteDataOp<NPUReg40XX::RegField_permutationType>(
                mlir::cast<VPUIPDPU::ODUPermuteDataOp>(op), initValues);
    }
};

using RegistersODUSparsityOp = struct RegistersODUSparsityOp {
    using TRegField_sp_valueType = NPUReg40XX::RegField_sp_valueType;
    using TRegField_sp_out_enType = NPUReg40XX::RegField_sp_out_enType;
};

class LowerToRegODUSparsityOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegODUSparsityOpInterfaceModel,
                                                               VPUIPDPU::ODUSparsityOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegODUSparsityOp<RegistersODUSparsityOp>(mlir::cast<VPUIPDPU::ODUSparsityOp>(op),
                                                                            initValues);
    }
};

class LowerToRegODUSwizzleDataOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegODUSwizzleDataOpInterfaceModel,
                                                               VPUIPDPU::ODUSwizzleDataOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegODUSwizzleDataOp<NPUReg40XX::RegField_swizzle_keyType>(
                mlir::cast<VPUIPDPU::ODUSwizzleDataOp>(op), initValues);
    }
};

class LowerToRegODUOutActivationsOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegODUOutActivationsOpInterfaceModel,
                                                               VPUIPDPU::ODUOutActivationsOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegODUOutActivationsOp<NPUReg40XX::RegField_dtypeType>(
                mlir::cast<VPUIPDPU::ODUOutActivationsOp>(op), initValues);
    }
};

class LowerToRegODUMemoryModeOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegODUMemoryModeOpInterfaceModel,
                                                               VPUIPDPU::ODUMemoryModeOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegODUMemoryModeOp<NPUReg40XX::RegField_modeType>(
                mlir::cast<VPUIPDPU::ODUMemoryModeOp>(op), initValues);
    }
};

class LowerToRegODUCmxPortsOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegODUCmxPortsOpInterfaceModel,
                                                               VPUIPDPU::ODUCmxPortsOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegODUCmxPortsOp<NPUReg40XX::RegField_cmx_port_muxing_disableType>(
                mlir::cast<VPUIPDPU::ODUCmxPortsOp>(op), initValues);
    }
};

using RegistersODUWriteCombineBufferOp = struct RegistersODUWriteCombineBufferOp {
    using TRegField_wcb_ac_modeType = NPUReg40XX::RegField_wcb_ac_modeType;
    using TRegField_wcb_sp_modeType = NPUReg40XX::RegField_wcb_sp_modeType;
};

class LowerToRegODUWriteCombineBufferOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegODUWriteCombineBufferOpInterfaceModel,
                                                               VPUIPDPU::ODUWriteCombineBufferOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegODUWriteCombineBufferOp<RegistersODUWriteCombineBufferOp>(
                mlir::cast<VPUIPDPU::ODUWriteCombineBufferOp>(op), initValues);
    }
};

using RegistersODUOutSubtensorOp = struct RegistersOduOutSubtensorOp {
    using TRegField_te_beg_xType = NPUReg40XX::RegField_te_beg_xType;
    using TRegField_te_beg_yType = NPUReg40XX::RegField_te_beg_yType;
    using TRegField_te_beg_zType = NPUReg40XX::RegField_te_beg_zType;
    using TRegField_te_end_xType = NPUReg40XX::RegField_te_end_xType;
    using TRegField_te_end_yType = NPUReg40XX::RegField_te_end_yType;
    using TRegField_te_end_zType = NPUReg40XX::RegField_te_end_zType;
};

class LowerToRegODUOutSubtensorOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegODUOutSubtensorOpInterfaceModel,
                                                               VPUIPDPU::ODUOutSubtensorOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegODUOutSubtensorOp<RegistersODUOutSubtensorOp>(
                mlir::cast<VPUIPDPU::ODUOutSubtensorOp>(op), initValues);
    }
};

using RegistersODUHaloCfgOp = struct RegistersODUHaloCfgOp {
    using TRegField_begin_xType = NPUReg40XX::RegField_begin_xType;
    using TRegField_begin_yType = NPUReg40XX::RegField_begin_yType;
    using TRegField_end_xType = NPUReg40XX::RegField_end_xType;
    using TRegField_ac_adr_offsetType = NPUReg40XX::RegField_ac_adr_offsetType;
    using TRegField_target_width_lsbType = NPUReg40XX::RegField_target_width_lsbType;
    using TRegField_target_width_msbType = NPUReg40XX::RegField_target_width_msbType;
    using TRegField_tile_selectType = NPUReg40XX::RegField_tile_selectType;
    using TRegField_sp_adr_offsetType = NPUReg40XX::RegField_sp_adr_offsetType;
};

class LowerToRegODUHaloCfgOpInterfaceModel final :
        public VPUIPDPU::LowerToNPURegInterface::ExternalModel<LowerToRegODUHaloCfgOpInterfaceModel,
                                                               VPUIPDPU::ODUHaloCfgOp> {
public:
    void lowerToRegisters(
            mlir::Operation* op,
            std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) const {
        VPUIPDPU::arch40xx::lowerToRegODUHaloCfgOp<RegistersODUHaloCfgOp>(mlir::cast<VPUIPDPU::ODUHaloCfgOp>(op),
                                                                          initValues);
    }
};

}  // namespace

void vpux::VPUIPDPU::arch40xx::registerLowerToRegistersInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPUIPDPU::VPUIPDPUDialect*) {
        // IDU ops:
        VPUIPDPU::IDUStorageElementOp::attachInterface<LowerToRegIDUStorageElementOpInterfaceModel>(*ctx);
        VPUIPDPU::IDUKernelOp::attachInterface<LowerToRegIDUKernelOpInterfaceModel>(*ctx);
        VPUIPDPU::IDUStrideOp::attachInterface<LowerToRegIDUStrideOpInterfaceModel>(*ctx);
        VPUIPDPU::IDUInActivationsOp::attachInterface<LowerToRegIDUInActivationsOpInterfaceModel>(*ctx);
        VPUIPDPU::IDUInputLayerCfgOp::attachInterface<LowerToRegIDUInputLayerCfgOpInterfaceModel>(*ctx);
        VPUIPDPU::IDUWeightsOp::attachInterface<LowerToRegIDUWeightsOpInterfaceModel>(*ctx);
        VPUIPDPU::IDUWorkloadCfgOp::attachInterface<LowerToRegIDUWorkloadCfgOpInterfaceModel>(*ctx);
        VPUIPDPU::IDUDepthWiseCfgOp::attachInterface<LowerToRegIDUDepthWiseCfgOpInterfaceModel>(*ctx);
        VPUIPDPU::IDUEltWiseCfgOp::attachInterface<LowerToRegIDUEltWiseCfgOpInterfaceModel>(*ctx);
        VPUIPDPU::IDUActSwizzleOp::attachInterface<LowerToRegIDUActSwizzleOpInterfaceModel>(*ctx);
        VPUIPDPU::IDUWeightSwizzleOp::attachInterface<LowerToRegIDUWeightSwizzleOpInterfaceModel>(*ctx);
        VPUIPDPU::IDUNthwNtkOp::attachInterface<LowerToRegIDUNthwNtkOpInterfaceModel>(*ctx);
        VPUIPDPU::IDUWorkloadSetOp::attachInterface<LowerToRegIDUWorkloadSetOpInterfaceModel>(*ctx);
        VPUIPDPU::IDUPaddingOp::attachInterface<LowerToRegIDUPaddingOpInterfaceModel>(*ctx);
        VPUIPDPU::IDUWeightSetOp::attachInterface<LowerToRegIDUWeightSetOpInterfaceModel>(*ctx);
        // MPE ops:
        VPUIPDPU::MPEActivationBiasOp::attachInterface<LowerToRegMpeActivationBiasOpInterfaceModel>(*ctx);
        VPUIPDPU::MPEWeightsBiasOp::attachInterface<LowerToRegMpeWeightsBiasOpInterfaceModel>(*ctx);
        // PPE ops:
        VPUIPDPU::PPEFpBiasAddOp::attachInterface<LowerToRegPPEFpBiasAddOpInterfaceModel>(*ctx);
        VPUIPDPU::PPEFpScalePreluMultOp::attachInterface<LowerToRegPPEFpScalePreluMultOpInterfaceModel>(*ctx);
        VPUIPDPU::PPEFpAddMultBypassOp::attachInterface<LowerToRegPPEFpAddMultBypassOpInterfaceModel>(*ctx);
        VPUIPDPU::PPEFpConvertOp::attachInterface<LowerToRegPPEFpConvertOpInterfaceModel>(*ctx);
        VPUIPDPU::PPEIntBiasAddOp::attachInterface<LowerToRegPPEIntBiasAddOpInterfaceModel>(*ctx);
        VPUIPDPU::PPEIntScaleMultOp::attachInterface<LowerToRegPPEIntScaleMultOpInterfaceModel>(*ctx);
        VPUIPDPU::PPEIntPreluMultOp::attachInterface<LowerToRegPPEIntPreluMultOpInterfaceModel>(*ctx);
        VPUIPDPU::PPEIntScaleShiftOp::attachInterface<LowerToRegPPEIntScaleShiftOpInterfaceModel>(*ctx);
        VPUIPDPU::PPEIntPreluShiftOp::attachInterface<LowerToRegPPEIntPreluShiftOpInterfaceModel>(*ctx);
        VPUIPDPU::PPEIntRoundOp::attachInterface<LowerToRegPPEIntRoundOpInterfaceModel>(*ctx);
        VPUIPDPU::PPEIntZeroPointOffsetOp::attachInterface<LowerToRegPPEIntZeroPointOffsetOpInterfaceModel>(*ctx);
        VPUIPDPU::PPEIntClampOp::attachInterface<LowerToRegPPEIntClampOpInterfaceModel>(*ctx);
        VPUIPDPU::PPEIntConvertOp::attachInterface<LowerToRegPPEIntConvertOpInterfaceModel>(*ctx);
        // ODU ops:
        VPUIPDPU::ODUOutTensorSizeOp::attachInterface<LowerToRegODUOutTensorSizeOpInterfaceModel>(*ctx);
        VPUIPDPU::ODUDataReuseOp::attachInterface<LowerToRegODUDataReuseOpInterfaceModel>(*ctx);
        VPUIPDPU::ODUPermuteDataOp::attachInterface<LowerToRegODUPermuteDataOpInterfaceModel>(*ctx);
        VPUIPDPU::ODUSparsityOp::attachInterface<LowerToRegODUSparsityOpInterfaceModel>(*ctx);
        VPUIPDPU::ODUSwizzleDataOp::attachInterface<LowerToRegODUSwizzleDataOpInterfaceModel>(*ctx);
        VPUIPDPU::ODUOutActivationsOp::attachInterface<LowerToRegODUOutActivationsOpInterfaceModel>(*ctx);
        VPUIPDPU::ODUMemoryModeOp::attachInterface<LowerToRegODUMemoryModeOpInterfaceModel>(*ctx);
        VPUIPDPU::ODUCmxPortsOp::attachInterface<LowerToRegODUCmxPortsOpInterfaceModel>(*ctx);
        VPUIPDPU::ODUWriteCombineBufferOp::attachInterface<LowerToRegODUWriteCombineBufferOpInterfaceModel>(*ctx);
        VPUIPDPU::ODUOutSubtensorOp::attachInterface<LowerToRegODUOutSubtensorOpInterfaceModel>(*ctx);
        VPUIPDPU::ODUHaloCfgOp::attachInterface<LowerToRegODUHaloCfgOpInterfaceModel>(*ctx);
    });
}
