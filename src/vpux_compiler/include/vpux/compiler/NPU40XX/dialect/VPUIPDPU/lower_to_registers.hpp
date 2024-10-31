//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux::VPURegMapped;
using namespace npu40xx;

namespace vpux::VPUIPDPU::arch40xx {

template <typename REG_TYPE, typename Type_Accepted_Registers>
uint64_t getTensorMode(mlir::Type type) {
    static_assert(
            std::is_same<REG_TYPE, typename Type_Accepted_Registers::TRegField_amodeType>::value ||
                    std::is_same<REG_TYPE, typename Type_Accepted_Registers::TRegField_wmodeType>::value ||
                    std::is_same<REG_TYPE,
                                 typename Type_Accepted_Registers::TRegField_dma_acc_info_compress_dtypeType>::value ||
                    std::is_same<REG_TYPE,
                                 typename Type_Accepted_Registers::TRegField_dma_acc_info_decompress_dtypeType>::value,
            "getTensorMode: Unsupported template argument REG_TYPE");

    if (auto quantized = type.dyn_cast<mlir::quant::QuantizedType>()) {
        return getTensorMode<REG_TYPE, Type_Accepted_Registers>(quantized.getStorageType());
    }
    if (std::is_same<REG_TYPE, typename Type_Accepted_Registers::TRegField_amodeType>::value ||
        std::is_same<REG_TYPE, typename Type_Accepted_Registers::TRegField_wmodeType>::value) {
        if (type.isF16()) {
            return static_cast<uint64_t>(nn_public::VpuInputTensorDType::FP16);
        } else if (type.isUnsignedInteger(8)) {
            return static_cast<uint64_t>(nn_public::VpuInputTensorDType::U8);
        } else if (type.isInteger(8)) {
            return static_cast<uint64_t>(nn_public::VpuInputTensorDType::I8);
        } else if (type.isInteger(4)) {
            return static_cast<uint64_t>(nn_public::VpuInputTensorDType::I4);
        } else if (type.isInteger(2)) {
            return static_cast<uint64_t>(nn_public::VpuInputTensorDType::I2);
        } else if (type.isBF16()) {
            return static_cast<uint64_t>(nn_public::VpuInputTensorDType::BF16);
        } else if (type.isUnsignedInteger(4)) {
            return static_cast<uint64_t>(nn_public::VpuInputTensorDType::U4);
        } else if (type.isInteger(1)) {
            return static_cast<uint64_t>(nn_public::VpuInputTensorDType::BIN);
        } else if (type.isFloat8E5M2()) {
            return static_cast<uint64_t>(nn_public::VpuInputTensorDType::FP8);
        } else if (type.isFloat8E4M3FN()) {
            return static_cast<uint64_t>(nn_public::VpuInputTensorDType::RESERVED);
        }
        VPUX_THROW("Invalid tensor type for DPU configuration {0}", type);
    } else if (std::is_same<REG_TYPE,
                            typename Type_Accepted_Registers::TRegField_dma_acc_info_compress_dtypeType>::value ||
               std::is_same<REG_TYPE,
                            typename Type_Accepted_Registers::TRegField_dma_acc_info_decompress_dtypeType>::value) {
        if (type.isUnsignedInteger(8) || type.isInteger(8) || type.isUnsignedInteger(4) || type.isInteger(4)) {
            return DMA_ACC_DTYPE_INT8_UINT8;
        } else if (type.isBF16() || type.isF16()) {
            return DMA_ACC_DTYPE_FP16_BF16;
        }
        VPUX_THROW("Invalid tensor type for DMA Acceleration configuration {0}", type);
    }
}

template <typename Type_Registers>
void lowerToRegIDUStorageElementOp(
        VPUIPDPU::IDUStorageElementOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    uint32_t seSize = op.getSeSize();

    // TODO: refactor the code in the if-else below (properly define hard-coded values) - E#82002
    if ((seSize != 0) && ((seSize & (seSize - 1)) == 0)) {  // seSize power of 2
        uint32_t seSizeHW = 0;

        // adjust SE size to HW supported limits
        if (seSize < 16) {
            seSize = 16;
        } else if (seSize > 8192) {
            // storage_element_size bigger than 8192, HW value adjusted for 8192;
            seSize = 8192;
        }

        while (seSize >>= 1) {
            ++seSizeHW;
        }  // seSizeHW = log2(seSize)

        // HW register NCE_DPU_Z_CONFIG.se_z_split has values: 1=16, 2=32....9=4096, 0=8192
        if (seSizeHW < 13) {
            seSizeHW -= 3;
        } else {
            seSizeHW = 0;
        }
        auto seSizeZSplit = checked_cast_reg<typename Type_Registers::TRegField_se_z_splitType>(seSizeHW);
        VPURegMapped::updateRegMappedInitializationValues(initValues, {{"z_config", {{"se_z_split", seSizeZSplit}}}});
    } else {
        auto seSizeHW = ((seSize + 15) >> 4) - 1;
        auto nonPowerOf2SESize = checked_cast_reg<typename Type_Registers::TRegField_npo2_se_sizeType>(seSizeHW);
        VPURegMapped::updateRegMappedInitializationValues(
                initValues,
                {{"z_config", {{"npo2_se_z_split_en", 1}}}, {"tensor_size1", {{"npo2_se_size", nonPowerOf2SESize}}}});
    }

    if (op.getNumSesInZDir().has_value()) {
        auto numSesInZDir =
                checked_cast_reg<typename Type_Registers::TRegField_num_ses_in_z_dirType>(op.getNumSesInZDir().value());
        VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                          {{"z_config", {{"num_ses_in_z_dir", numSesInZDir}}}});
    }
}

template <typename Type_Registers>
void lowerToRegIDUKernelOp(
        VPUIPDPU::IDUKernelOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto kernelX = checked_cast_reg<typename Type_Registers::TRegField_kernel_xType>(op.getKernelX());
    auto kernelY = checked_cast_reg<typename Type_Registers::TRegField_kernel_yType>(op.getKernelY());
    VPURegMapped::updateRegMappedInitializationValues(
            initValues, {{"kernel_pad_cfg", {{"kernel_y", kernelY}, {"kernel_x", kernelX}}}});
}

template <typename Type_Registers>
void lowerToRegIDUStrideOp(
        VPUIPDPU::IDUStrideOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto strideY = checked_cast_reg<typename Type_Registers::TRegField_stride_yType>(op.getStrideY() - 1);
    auto strideX = checked_cast_reg<typename Type_Registers::TRegField_strideType>(op.getStrideX() - 1);
    if (op.getStrideY() == op.getStrideX()) {
        VPURegMapped::updateRegMappedInitializationValues(initValues, {{"tensor_mode", {{"stride", strideX}}}});
    } else {
        VPURegMapped::updateRegMappedInitializationValues(
                initValues, {{"kernel_pad_cfg", {{"stride_y", strideY}, {"stride_y_en", 1}}},
                             {"tensor_mode", {{"stride", strideX}}}});
    }
}

template <typename Type_Registers>
void lowerToRegIDUInActivationsOp(
        VPUIPDPU::IDUInActivationsOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto inActivations = op.getInActivations();
    auto inActivationsType = inActivations.getType().cast<vpux::NDTypeInterface>().getElementType();
    auto inActivationShape = getShape(inActivations);
    const auto dimY =
            checked_cast_reg<typename Type_Registers::TRegField_tensor_size_yType>(inActivationShape[Dims4D::Act::H]);
    const auto dimX =
            checked_cast_reg<typename Type_Registers::TRegField_tensor_size_xType>(inActivationShape[Dims4D::Act::W]);
    const auto dimZ =
            checked_cast_reg<typename Type_Registers::TRegField_tensor_size_zType>(inActivationShape[Dims4D::Act::C]);
    auto tensorMode = checked_cast_reg<typename Type_Registers::TRegField_amodeType>(
            getTensorMode<typename Type_Registers::TRegField_amodeType,
                          typename Type_Registers::TTensorModeAcceptedRegisters>(inActivationsType));
    auto actDense = checked_cast_reg<typename Type_Registers::TRegField_act_denseType>(!op.getInSparse());

    VPURegMapped::updateRegMappedInitializationValues(
            initValues, {{"tensor_size0", {{"tensor_size_x", dimX}, {"tensor_size_y", dimY}}},
                         {"tensor_size1", {{"tensor_size_z", dimZ}}},
                         {"tensor_mode", {{"amode", tensorMode}}},
                         {"kernel_pad_cfg", {{"act_dense", actDense}}}});
}

template <typename Type_Registers>
void lowerToRegIDUInputLayerCfgOp(
        VPUIPDPU::IDUInputLayerCfgOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto sparsityPattern =
            checked_cast_reg<typename Type_Registers::TRegField_cm_sp_patternType>(op.getSparsityPattern());
    auto inputCompressed =
            checked_cast_reg<typename Type_Registers::TRegField_layer1_cmp_enType>(op.getInputCompressed());
    VPURegMapped::updateRegMappedInitializationValues(
            initValues,
            {{"z_config", {{"cm_sp_pattern", sparsityPattern}}},
             {"kernel_pad_cfg",
              {{"act_dense", 1}, {"wt_dense", 1}, {"layer1_wt_sp_ins", 1}, {"layer1_cmp_en", inputCompressed}}},
             {"tensor_size1", {{"tensor_size_z", 16}}}});
}

template <typename Type_Registers>
void lowerToRegIDUWeightsOp(
        VPUIPDPU::IDUWeightsOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto wmode = checked_cast_reg<typename Type_Registers::TRegField_wmodeType>(
            getTensorMode<typename Type_Registers::TRegField_wmodeType,
                          typename Type_Registers::TTensorModeAcceptedRegisters>(op.getWmode()));
    auto wtDense = checked_cast_reg<typename Type_Registers::TRegField_wt_denseType>(!op.getWtSparse());
    VPURegMapped::updateRegMappedInitializationValues(
            initValues, {{"tensor_mode", {{"wmode", wmode}}}, {"kernel_pad_cfg", {{"wt_dense", wtDense}}}});
    if (op.getPoolWtData().has_value()) {
        auto poolWtData =
                checked_cast_reg<typename Type_Registers::TRegField_pool_wt_dataType>(op.getPoolWtData().value());
        VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                          {{"elops_wload", {{"pool_wt_data", poolWtData}}}});
    }
}

bool lowerToRegIDUWorkloadCfgOp(
        VPUIPDPU::IDUWorkloadCfgOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues);

template <typename TRegField_dw_opt_offsetType>
void lowerToRegIDUDepthWiseCfgOp(
        VPUIPDPU::IDUDepthWiseCfgOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    if (op.getDw_3x3s1OptDis()) {
        VPURegMapped::updateRegMappedInitializationValues(initValues, {{"base_offset_b", {{"dw_3x3s1_opt_dis", 1}}}});
    }
    if (op.getDwOptOffset().has_value()) {
        auto dwOptOffset = checked_cast_reg<TRegField_dw_opt_offsetType>(op.getDwOptOffset().value());
        VPURegMapped::updateRegMappedInitializationValues(
                initValues, {{"base_offset_b", {{"dw_opt_en", 1}, {"dw_opt_offset", dwOptOffset}}},
                             {"kernel_pad_cfg", {{"pool_opt_en", 1}}}});
    }
}

template <typename Type_Registers>
void lowerToRegIDUEltWiseCfgOp(
        VPUIPDPU::IDUEltWiseCfgOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto elopScaleA = checked_cast_reg<typename Type_Registers::TRegField_elop_scale_aType>(
            op.getElopScaleAAttr().dyn_cast<mlir::IntegerAttr>().getInt());
    auto elopScaleB = checked_cast_reg<typename Type_Registers::TRegField_elop_scale_bType>(
            op.getElopScaleBAttr().dyn_cast<mlir::IntegerAttr>().getInt());
    VPURegMapped::updateRegMappedInitializationValues(
            initValues, {{"elop_scale", {{"elop_scale_a", elopScaleA}, {"elop_scale_b", elopScaleB}}}});
}

template <typename TRegField_swizzle_key_offsetType>
void lowerToRegIDUActSwizzleOp(
        VPUIPDPU::IDUActSwizzleOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto swizzleKey = checked_cast_reg<TRegField_swizzle_key_offsetType>(op.getSwizzleKey());
    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                      {
                                                              {"offset_addr", {{"swizzle_key_offset", swizzleKey}}},
                                                      });
}

template <typename TRegField_wt_swizzle_keyType>
void lowerToRegIDUWeightSwizzleOp(
        VPUIPDPU::IDUWeightSwizzleOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto wtSwizzleKey = checked_cast_reg<TRegField_wt_swizzle_keyType>(op.getWtSwizzleKey());
    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                      {{"offset_addr", {{"wt_swizzle_key", wtSwizzleKey}}}});
}

template <typename TRegField_nthw_ntkType>
void lowerToRegIDUNthwNtkOp(
        VPUIPDPU::IDUNthwNtkOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto nthwNtk = checked_cast_reg<TRegField_nthw_ntkType>(op.getNthwNtk());
    VPURegMapped::updateRegMappedInitializationValues(initValues, {
                                                                          {"offset_addr", {{"nthw_ntk", nthwNtk}}},
                                                                  });
}

template <typename Type_Registers>
void lowerToRegIDUWorkloadSetOp(
        VPUIPDPU::IDUWorkloadSetOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto startX = checked_cast_reg<typename Type_Registers::TRegField_workload_start_xType>(op.getStartX());
    auto startY = checked_cast_reg<typename Type_Registers::TRegField_workload_start_yType>(op.getStartY());
    auto startZ = checked_cast_reg<typename Type_Registers::TRegField_workload_start_zType>(op.getStartZ());
    auto sizeX = checked_cast_reg<typename Type_Registers::TRegField_workload_size_xType>(op.getSizeX());
    auto sizeY = checked_cast_reg<typename Type_Registers::TRegField_workload_size_yType>(op.getSizeY());
    auto sizeZ = checked_cast_reg<typename Type_Registers::TRegField_workload_size_zType>(op.getSizeZ());

    VPURegMapped::updateRegMappedInitializationValues(
            initValues, {{"workload_start0", {{"workload_start_x", startX}, {"workload_start_y", startY}}},
                         {"workload_start1", {{"workload_start_z", startZ}}},
                         {"workload_size0", {{"workload_size_x", sizeX}, {"workload_size_y", sizeY}}},
                         {"workload_size1", {{"workload_size_z", sizeZ}}}});
}

template <typename Type_Registers>
void lowerToRegIDUPaddingOp(
        VPUIPDPU::IDUPaddingOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto padUp =
            checked_cast_reg<typename Type_Registers::TRegField_pad_count_upType>(op.getPadCount().getTop().getInt());
    auto padLeft = checked_cast_reg<typename Type_Registers::TRegField_pad_count_leftType>(
            op.getPadCount().getLeft().getInt());
    auto padDown = checked_cast_reg<typename Type_Registers::TRegField_pad_count_downType>(
            op.getPadCount().getBottom().getInt());
    auto padRight = checked_cast_reg<typename Type_Registers::TRegField_pad_count_rightType>(
            op.getPadCount().getRight().getInt());
    VPURegMapped::updateRegMappedInitializationValues(initValues, {
                                                                          {"workload_size1",
                                                                           {{"pad_count_up", padUp},
                                                                            {"pad_count_left", padLeft},
                                                                            {"pad_count_down", padDown},
                                                                            {"pad_count_right", padRight}}},
                                                                  });
}

template <typename Type_Registers>
void lowerToRegIDUWeightSetOp(
        VPUIPDPU::IDUWeightSetOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto weightsStart = checked_cast_reg<typename Type_Registers::TRegField_weight_startType>(op.getWeightStart());
    auto weightsNum = checked_cast_reg<typename Type_Registers::TRegField_weight_numType>(op.getWeightNum());
    auto weightsSize = checked_cast_reg<typename Type_Registers::TRegField_weight_sizeType>(op.getWeightSize());

    // weight_start register will be modified by relocation mechanism based on provided offset info
    VPURegMapped::updateRegMappedInitializationValues(initValues, {{"weight_size", {{"weight_size", weightsSize}}},
                                                                   {"weight_num", {{"weight_num", weightsNum}}},
                                                                   {"weight_start", {{"weight_start", weightsStart}}}});
}

template <typename TRegField_mpe_actbiasType>
void lowerToRegMPEActivationBiasOp(
        VPUIPDPU::MPEActivationBiasOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto actBias = checked_cast_reg<TRegField_mpe_actbiasType>(op.getActBias());
    VPURegMapped::updateRegMappedInitializationValues(initValues, {{"mpe_cfg", {{"mpe_actbias", actBias}}}});
}

template <typename TRegField_mpe_wtbiasType>
void lowerToRegMPEWeightsBiasOp(
        VPUIPDPU::MPEWeightsBiasOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto wtBias = checked_cast_reg<TRegField_mpe_wtbiasType>(op.getWeightsBias());
    VPURegMapped::updateRegMappedInitializationValues(initValues, {{"mpe_cfg", {{"mpe_wtbias", wtBias}}}});
}

template <typename TRegField_ppe_fp_biasType>
void lowerToRegPPEFpBiasAddOp(
        VPUIPDPU::PPEFpBiasAddOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    VPUX_THROW_UNLESS((op.getScaleTable() != nullptr) ^ op.getBiasStatic().has_value(),
                      "op {0} has ambiguous parameters", op);
    if (op.getScaleTable() != nullptr) {
        VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                          {
                                                                  {"ppe_scale_ctrl", {{"ppe_fp_scale_override", 0}}},
                                                          });
    }
    if (op.getBiasStatic().has_value()) {
        uint64_t biasStatic = checked_cast_reg<TRegField_ppe_fp_biasType>(op.getBiasStatic().value().convertToFloat());
        VPURegMapped::updateRegMappedInitializationValues(
                initValues,
                {{"ppe_scale_ctrl", {{"ppe_fp_scale_override", 1}}}, {"ppe_fp_bias", {{"ppe_fp_bias", biasStatic}}}});
    }
}

template <typename Type_Registers>
void lowerToRegPPEFpScalePreluMultOp(
        VPUIPDPU::PPEFpScalePreluMultOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    VPUX_THROW_UNLESS((op.getScaleTable() != nullptr) ^ op.getScaleStatic().has_value(),
                      "op {0} has ambiguous parameters", op);
    if (op.getScaleTable()) {
        VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                          {
                                                                  {"ppe_scale_ctrl", {{"ppe_fp_scale_override", 0}}},
                                                          });
    }
    if (op.getScaleStatic().has_value()) {
        uint64_t scaleStatic = checked_cast_reg<typename Type_Registers::TRegField_ppe_fp_scaleType>(
                op.getScaleStatic().value().convertToFloat());
        VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                          {{"ppe_scale_ctrl", {{"ppe_fp_scale_override", 1}}},
                                                           {"ppe_fp_scale", {{"ppe_fp_scale", scaleStatic}}}});
    }
    if (op.getPreluAlpha().has_value()) {
        uint64_t preluAlpha = checked_cast_reg<typename Type_Registers::TRegField_ppe_fp_preluType>(
                op.getPreluAlpha().value().convertToFloat());
        VPURegMapped::updateRegMappedInitializationValues(
                initValues,
                {{"ppe_fp_cfg", {{"ppe_fp_prelu_en", 1}}}, {"ppe_fp_prelu", {{"ppe_fp_prelu", preluAlpha}}}});
    }
}

template <typename TRegField_ppe_fp_bypassType>
void lowerToRegPPEFpAddMultBypassOp(
        VPUIPDPU::PPEFpAddMultBypassOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto bypassMode = checked_cast_reg<TRegField_ppe_fp_bypassType>(op.getBypassMode());
    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                      {
                                                              {"ppe_fp_cfg", {{"ppe_fp_bypass", bypassMode}}},
                                                      });
}

template <typename Type_Registers>
void lowerToRegPPEFpScalePreluMultOp(
        VPUIPDPU::PPEFpConvertOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto convertMode = checked_cast_reg<typename Type_Registers::TRegField_ppe_fp_convertType>(op.getConvertMode());
    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                      {
                                                              {"ppe_fp_cfg", {{"ppe_fp_convert", convertMode}}},
                                                      });
    if (op.getClampMode().has_value()) {
        auto clampMode =
                checked_cast_reg<typename Type_Registers::TRegField_ppe_fp16_clampType>(op.getClampMode().value());
        VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                          {
                                                                  {"ppe_misc", {{"ppe_fp16_clamp", clampMode}}},
                                                          });
    }
    if (op.getFtzMode().has_value()) {
        auto ftzMode = checked_cast_reg<typename Type_Registers::TRegField_ppe_fp16_ftzType>(op.getFtzMode().value());
        VPURegMapped::updateRegMappedInitializationValues(initValues, {
                                                                              {"ppe_misc", {{"ppe_fp16_ftz", ftzMode}}},
                                                                      });
    }
    if (op.getBf16RoundMode().has_value()) {
        auto bf16RoundMode =
                checked_cast_reg<typename Type_Registers::TRegField_ppe_bf16_roundType>(op.getBf16RoundMode().value());
        VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                          {
                                                                  {"ppe_fp_cfg", {{"ppe_bf16_round", bf16RoundMode}}},
                                                          });
    }
}

template <typename TRegField_ppe_biasType>
void lowerToRegPPEIntBiasAddOp(
        VPUIPDPU::PPEIntBiasAddOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    VPUX_THROW_UNLESS((op.getScaleTable() != nullptr) ^ op.getBiasStatic().has_value(),
                      "op {0} has ambiguous parameters", op);
    if (op.getScaleTable() != nullptr) {
        VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                          {
                                                                  {"ppe_scale_ctrl", {{"ppe_scale_override", 0}}},
                                                          });
    }
    if (op.getBiasStatic().has_value()) {
        uint64_t biasStatic = checked_cast_reg<TRegField_ppe_biasType>(op.getBiasStatic().value());
        VPURegMapped::updateRegMappedInitializationValues(initValues, {{"ppe_scale_ctrl", {{"ppe_scale_override", 1}}},
                                                                       {"ppe_bias", {{"ppe_bias", biasStatic}}}});
    }
}

template <typename TRegField_ppe_scale_multType>
void lowerToRegPPEIntScaleMultOp(
        VPUIPDPU::PPEIntScaleMultOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    VPUX_THROW_UNLESS((op.getScaleTable() != nullptr) ^ op.getScaleStatic().has_value(),
                      "op {0} has ambiguous parameters", op);
    if (op.getScaleTable()) {
        VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                          {
                                                                  {"ppe_scale_ctrl", {{"ppe_scale_override", 0}}},
                                                          });
    }
    if (op.getScaleStatic().has_value()) {
        uint64_t scaleStatic = checked_cast_reg<TRegField_ppe_scale_multType>(op.getScaleStatic().value());
        VPURegMapped::updateRegMappedInitializationValues(
                initValues,
                {{"ppe_scale_ctrl", {{"ppe_scale_override", 1}}}, {"ppe_scale", {{"ppe_scale_mult", scaleStatic}}}});
    }
}

template <typename TRegField_ppe_prelu_multType>
void lowerToRegPPEIntPreluMultOp(
        VPUIPDPU::PPEIntPreluMultOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto preluMultStatic = checked_cast_reg<TRegField_ppe_prelu_multType>(op.getPreluMultStatic());
    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                      {
                                                              {"ppe_prelu", {{"ppe_prelu_mult", preluMultStatic}}},
                                                      });
}

template <typename TRegField_ppe_scale_shiftType>
void lowerToRegPPEIntScaleShiftOp(
        VPUIPDPU::PPEIntScaleShiftOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    VPUX_THROW_UNLESS((op.getScaleTable() != nullptr) ^ op.getShiftStatic().has_value(),
                      "op {0} has ambiguous parameters", op);
    if (op.getScaleTable() != nullptr) {
        VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                          {
                                                                  {"ppe_scale_ctrl", {{"ppe_scale_override", 0}}},
                                                          });
    }
    if (op.getShiftStatic().has_value()) {
        uint64_t shiftStatic = checked_cast_reg<TRegField_ppe_scale_shiftType>(op.getShiftStatic().value());
        VPURegMapped::updateRegMappedInitializationValues(
                initValues,
                {{"ppe_scale_ctrl", {{"ppe_scale_override", 1}}}, {"ppe_scale", {{"ppe_scale_shift", shiftStatic}}}});
    }
}

template <typename TRegField_ppe_prelu_shiftType>
void lowerToRegPPEIntPreluShiftOp(
        VPUIPDPU::PPEIntPreluShiftOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto preluShiftStatic = checked_cast_reg<TRegField_ppe_prelu_shiftType>(op.getPreluShiftStatic());
    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                      {
                                                              {"ppe_prelu", {{"ppe_prelu_shift", preluShiftStatic}}},
                                                      });
}

template <typename TRegField_ppe_scale_roundType>
void lowerToRegPPEIntRoundOp(
        VPUIPDPU::PPEIntRoundOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto roundMode = checked_cast_reg<TRegField_ppe_scale_roundType>(op.getRoundMode());
    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                      {
                                                              {"ppe_scale", {{"ppe_scale_round", roundMode}}},
                                                      });
}

template <typename TRegField_ppe_g8_bias_cType>
void lowerToRegPPEIntZeroPointOffsetOp(
        VPUIPDPU::PPEIntZeroPointOffsetOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto zeroPointStatic = checked_cast_reg<TRegField_ppe_g8_bias_cType>(op.getZeroPointStatic());
    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                      {
                                                              {"ppe_cfg", {{"ppe_g8_bias_c", zeroPointStatic}}},
                                                      });
}

void lowerToRegPPEIntClampOp(
        VPUIPDPU::PPEIntClampOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues);

template <typename TRegField_ppe_i32_convertType>
void lowerToRegPPEIntConvertOp(
        VPUIPDPU::PPEIntConvertOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto convertMode = checked_cast_reg<TRegField_ppe_i32_convertType>(op.getConvertMode());
    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                      {
                                                              {"ppe_misc", {{"ppe_i32_convert", convertMode}}},
                                                      });
}

template <typename Type_Registers>
void lowerToRegODUOutTensorSizeOp(
        VPUIPDPU::ODUOutTensorSizeOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto xDim = checked_cast_reg<typename Type_Registers::TRegField_te_dim_xType>(op.getDimX() - 1);
    auto yDim = checked_cast_reg<typename Type_Registers::TRegField_te_dim_yType>(op.getDimY() - 1);
    auto zDim = checked_cast_reg<typename Type_Registers::TRegField_te_dim_zType>(op.getDimZ() - 1);
    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                      {
                                                              {"te_dim0", {{"te_dim_y", yDim}, {"te_dim_z", zDim}}},
                                                              {"te_dim1", {{"te_dim_x", xDim}}},
                                                      });
}

template <typename TRegField_nthwType>
void lowerToRegODUDataReuseOp(
        VPUIPDPU::ODUDataReuseOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto activationReuse = checked_cast_reg<TRegField_nthwType>(op.getActivationReuse());
    VPURegMapped::updateRegMappedInitializationValues(initValues, {
                                                                          {"odu_cfg", {{"nthw", activationReuse}}},
                                                                  });
}

template <typename TRegField_permutationType>
void lowerToRegODUPermuteDataOp(
        VPUIPDPU::ODUPermuteDataOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto permuteMode = checked_cast_reg<TRegField_permutationType>(op.getPermuteMode());
    VPURegMapped::updateRegMappedInitializationValues(initValues, {
                                                                          {"odu_cfg", {{"permutation", permuteMode}}},
                                                                  });
}

template <typename Type_Registers>
void lowerToRegODUSparsityOp(
        VPUIPDPU::ODUSparsityOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto sparseValue =
            checked_cast_reg<typename Type_Registers::TRegField_sp_valueType>(op.getSparseValue().value_or(0));
    auto compressionEnabled = checked_cast_reg<typename Type_Registers::TRegField_sp_out_enType>(
            op.getCompressionEnabled().value_or(true));
    uint64_t writeSp = op.getSparsityMap() ? 1 : 0;
    VPURegMapped::updateRegMappedInitializationValues(
            initValues,
            {
                    {"odu_cfg", {{"sp_value", sparseValue}, {"sp_out_en", compressionEnabled}, {"write_sp", writeSp}}},
            });
}

template <typename TRegField_swizzle_keyType>
void lowerToRegODUSwizzleDataOp(
        VPUIPDPU::ODUSwizzleDataOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto swizzleKey = checked_cast_reg<TRegField_swizzle_keyType>(op.getSwizzleKey());
    VPURegMapped::updateRegMappedInitializationValues(initValues, {
                                                                          {"odu_cfg", {{"swizzle_key", swizzleKey}}},
                                                                  });
}

template <typename TRegField_dtypeType>
void lowerToRegODUOutActivationsOp(
        VPUIPDPU::ODUOutActivationsOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    uint64_t dataWidth(0);
    if (op.getDataWidth().has_value()) {
        dataWidth = checked_cast_reg<TRegField_dtypeType>(op.getDataWidth().value());
    } else {
        auto outActType = op.getOutActivations().getType().cast<mlir::MemRefType>().getElementType();
        dataWidth = checked_cast_reg<TRegField_dtypeType>(getDataBitWidth(outActType));
    }

    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                      {
                                                              {"odu_cfg", {{"dtype", dataWidth}, {"write_ac", 1}}},
                                                      });
}

template <typename TRegField_modeType>
void lowerToRegODUMemoryModeOp(
        VPUIPDPU::ODUMemoryModeOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto memMode = checked_cast_reg<TRegField_modeType>(op.getMemMode());
    VPURegMapped::updateRegMappedInitializationValues(initValues, {
                                                                          {"odu_cfg", {{"mode", memMode}}},
                                                                  });
}

template <typename TRegField_cmx_port_muxing_disableType>
void lowerToRegODUCmxPortsOp(
        VPUIPDPU::ODUCmxPortsOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto cmxPorts = checked_cast_reg<TRegField_cmx_port_muxing_disableType>(op.getCmxPorts());
    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                      {
                                                              {"odu_cfg", {{"cmx_port_muxing_disable", cmxPorts}}},
                                                      });
}

template <typename Type_Registers>
void lowerToRegODUWriteCombineBufferOp(
        VPUIPDPU::ODUWriteCombineBufferOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto activationsMode =
            checked_cast_reg<typename Type_Registers::TRegField_wcb_ac_modeType>(op.getActivationsMode());
    auto sparsityMode = checked_cast_reg<typename Type_Registers::TRegField_wcb_sp_modeType>(
            op.getSparsityMode().value_or(vpux::VPUIPDPU::ODUWcbCombineMode::WCB_COMBINE_BY_CONTEXT));

    VPURegMapped::updateRegMappedInitializationValues(
            initValues,
            {
                    {"odu_cfg", {{"wcb_bypass", 0}, {"wcb_ac_mode", activationsMode}, {"wcb_sp_mode", sparsityMode}}},
            });
}

template <typename Type_Registers>
void lowerToRegODUOutSubtensorOp(
        VPUIPDPU::ODUOutSubtensorOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    auto begX = checked_cast_reg<typename Type_Registers::TRegField_te_beg_xType>(op.getBeginCoordX());
    auto begY = checked_cast_reg<typename Type_Registers::TRegField_te_beg_yType>(op.getBeginCoordY());
    auto begZ = checked_cast_reg<typename Type_Registers::TRegField_te_beg_zType>(op.getBeginCoordZ());
    auto endX = checked_cast_reg<typename Type_Registers::TRegField_te_end_xType>(op.getEndCoordX());
    auto endY = checked_cast_reg<typename Type_Registers::TRegField_te_end_yType>(op.getEndCoordY());
    auto endZ = checked_cast_reg<typename Type_Registers::TRegField_te_end_zType>(op.getEndCoordZ());
    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                      {
                                                              {"te_beg0", {{"te_beg_y", begY}, {"te_beg_z", begZ}}},
                                                              {"te_beg1", {{"te_beg_x", begX}}},
                                                              {"te_end0", {{"te_end_y", endY}, {"te_end_z", endZ}}},
                                                              {"te_end1", {{"te_end_x", endX}}},
                                                      });
}

template <typename Type_Registers>
void lowerToRegODUHaloCfgOp(
        VPUIPDPU::ODUHaloCfgOp op,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues) {
    uint8_t haloRegionIdx(0);
    for (const auto& haloRegionOp : op.getRegion().getOps()) {
        auto opHaloReg = mlir::dyn_cast_or_null<VPUIPDPU::ODUHaloRegionOp>(&haloRegionOp);
        if (opHaloReg == nullptr) {
            VPUX_THROW("Found invalid child op under ODUHaloCfgOp: {0}", haloRegionOp);
        }
        auto begX = checked_cast_reg<typename Type_Registers::TRegField_begin_xType>(opHaloReg.getBeginCoordX());
        auto begY = checked_cast_reg<typename Type_Registers::TRegField_begin_yType>(opHaloReg.getBeginCoordY());
        auto endX = checked_cast_reg<typename Type_Registers::TRegField_end_xType>(opHaloReg.getEndCoordX());
        auto endY = checked_cast_reg<typename Type_Registers::TRegField_end_xType>(opHaloReg.getEndCoordY());
        auto actOffset = checked_cast_reg<typename Type_Registers::TRegField_ac_adr_offsetType>(
                opHaloReg.getActivationsOffset());

        uint64_t lsbWidthValue(0), msbWidthValue(0);
        computeLsbAndMsbFromTargetWidth<typename Type_Registers::TRegField_target_width_lsbType,
                                        typename Type_Registers::TRegField_target_width_msbType>(
                opHaloReg.getTargetWidth(), msbWidthValue, lsbWidthValue);

        auto targetWidthLsb = checked_cast_reg<typename Type_Registers::TRegField_target_width_lsbType>(lsbWidthValue);
        auto targetWidthMsb = checked_cast_reg<typename Type_Registers::TRegField_target_width_msbType>(msbWidthValue);
        auto castToTile =
                checked_cast_reg<typename Type_Registers::TRegField_tile_selectType>(opHaloReg.getCastToTile());
        auto sparsityOffset = checked_cast_reg<typename Type_Registers::TRegField_sp_adr_offsetType>(
                opHaloReg.getSparsityOffset().value_or(0));

        auto haloRegionA = std::string("halo_region" + std::to_string(haloRegionIdx) + "A");
        auto haloRegionB = std::string("halo_region" + std::to_string(haloRegionIdx) + "B");
        auto haloRegionC = std::string("halo_region" + std::to_string(haloRegionIdx) + "C");
        auto haloRegionD = std::string("halo_region" + std::to_string(haloRegionIdx) + "D");

        VPURegMapped::updateRegMappedInitializationValues(
                initValues,
                {
                        {haloRegionA, {{"sp_adr_offset", sparsityOffset}, {"tile_select", castToTile}, {"enable", 1}}},
                        {haloRegionB, {{"ac_adr_offset", actOffset}, {"target_width_lsb", targetWidthLsb}}},
                        {haloRegionC, {{"begin_x", begX}, {"begin_y", begY}, {"target_width_msb", targetWidthMsb}}},
                        {haloRegionD, {{"end_x", endX}, {"end_y", endY}}},
                });
        haloRegionIdx++;
    }
}

void lowerToRegBarrierCfgOpWithDPUInvariantParent(
        VPUIPDPU::DPUInvariantOp& dpuInvariantOp,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues);

void lowerToRegBarrierCfgOpWithDPUVariantParent(
        VPUIPDPU::DPUVariantOp& dpuVariantOp,
        std::map<std::string, std::map<std::string, vpux::VPURegMapped::RegFieldValue>>& initValues);

}  // namespace vpux::VPUIPDPU::arch40xx
