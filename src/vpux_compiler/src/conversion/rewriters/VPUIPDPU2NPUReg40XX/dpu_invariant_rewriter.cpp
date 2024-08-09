//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUIPDPU2NPUReg40XX/dpu_invariant_rewriter.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"
#include "vpux/compiler/utils/traits_utils.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace vpux::VPURegMapped;
using namespace npu40xx;

namespace {

template <typename REG_TYPE>
uint64_t getTensorMode(mlir::Type type) {
    static_assert(std::is_same<REG_TYPE, NPUReg40XX::RegField_amodeType>::value ||
                          std::is_same<REG_TYPE, NPUReg40XX::RegField_wmodeType>::value ||
                          std::is_same<REG_TYPE, NPUReg40XX::RegField_dma_acc_info_compress_dtypeType>::value ||
                          std::is_same<REG_TYPE, NPUReg40XX::RegField_dma_acc_info_decompress_dtypeType>::value,
                  "getTensorMode: Unsupported template argument REG_TYPE");

    if (auto quantized = type.dyn_cast<mlir::quant::QuantizedType>()) {
        return getTensorMode<REG_TYPE>(quantized.getStorageType());
    }
    if (std::is_same<REG_TYPE, NPUReg40XX::RegField_amodeType>::value ||
        std::is_same<REG_TYPE, NPUReg40XX::RegField_wmodeType>::value) {
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
        }
        VPUX_THROW("Invalid tensor type for DPU configuration {0}", type);
    } else if (std::is_same<REG_TYPE, NPUReg40XX::RegField_dma_acc_info_compress_dtypeType>::value ||
               std::is_same<REG_TYPE, NPUReg40XX::RegField_dma_acc_info_decompress_dtypeType>::value) {
        if (type.isUnsignedInteger(8) || type.isInteger(8) || type.isUnsignedInteger(4) || type.isInteger(4)) {
            return DMA_ACC_DTYPE_INT8_UINT8;
        } else if (type.isBF16() || type.isF16()) {
            return DMA_ACC_DTYPE_FP16_BF16;
        }
        VPUX_THROW("Invalid tensor type for DMA Acceleration configuration {0}", type);
    }
}

VPUIPDPU::ODUDataBitWidth getDataBitWidth(mlir::Type outActType) {
    auto asIntegerType = outActType.dyn_cast<mlir::IntegerType>();
    auto asFloatType = outActType.dyn_cast<mlir::FloatType>();
    VPUX_THROW_UNLESS(asIntegerType || asFloatType, "Not a Float or Integer Type");

    const auto width = asIntegerType ? asIntegerType.getWidth() : asFloatType.getWidth();
    const auto oduBitWidth = VPUIPDPU::symbolizeODUDataBitWidth(log2(width));
    VPUX_THROW_UNLESS(oduBitWidth.has_value(), "Unable to determine data bit width from out_activations {0}",
                      outActType);

    return oduBitWidth.value();
}

}  // namespace

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

void DPUInvariantRewriter::fillIDUCfg(mlir::Region& DPURegion,
                                      std::map<std::string, std::map<std::string, uint64_t>>& initValues) const {
    auto IDUCfgOps = DPURegion.getOps<VPUIPDPU::IDUCfgOp>();
    if (!IDUCfgOps.empty()) {
        auto IDUCfgOp = *IDUCfgOps.begin();

        for (auto& IDUOp : IDUCfgOp.getRegion().getOps()) {
            if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUStorageElementOp>(&IDUOp)) {
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
                    auto seSizeZSplit = checked_cast_reg<NPUReg40XX::RegField_se_z_splitType>(seSizeHW);
                    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                                      {{"z_config", {{"se_z_split", seSizeZSplit}}}});
                } else {
                    auto seSizeHW = ((seSize + 15) >> 4) - 1;
                    auto nonPowerOf2SESize = checked_cast_reg<NPUReg40XX::RegField_npo2_se_sizeType>(seSizeHW);
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues, {{"z_config", {{"npo2_se_z_split_en", 1}}},
                                         {"tensor_size1", {{"npo2_se_size", nonPowerOf2SESize}}}});
                }

                if (op.getNumSesInZDir().has_value()) {
                    auto numSesInZDir =
                            checked_cast_reg<NPUReg40XX::RegField_num_ses_in_z_dirType>(op.getNumSesInZDir().value());
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues, {{"z_config", {{"num_ses_in_z_dir", numSesInZDir}}}});
                }

            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUKernelOp>(&IDUOp)) {
                auto kernelX = checked_cast_reg<NPUReg40XX::RegField_kernel_xType>(op.getKernelX());
                auto kernelY = checked_cast_reg<NPUReg40XX::RegField_kernel_yType>(op.getKernelY());
                VPURegMapped::updateRegMappedInitializationValues(
                        initValues, {{"kernel_pad_cfg", {{"kernel_y", kernelY}, {"kernel_x", kernelX}}}});
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUStrideOp>(&IDUOp)) {
                auto strideY = checked_cast_reg<NPUReg40XX::RegField_stride_yType>(op.getStrideY() - 1);
                auto strideX = checked_cast_reg<NPUReg40XX::RegField_strideType>(op.getStrideX() - 1);
                if (op.getStrideY() == op.getStrideX()) {
                    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                                      {{"tensor_mode", {{"stride", strideX}}}});
                } else {
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues, {{"kernel_pad_cfg", {{"stride_y", strideY}, {"stride_y_en", 1}}},
                                         {"tensor_mode", {{"stride", strideX}}}});
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUInActivationsOp>(&IDUOp)) {
                auto inActivations = op.getInActivations();
                auto inActivationsType = inActivations.getType().cast<vpux::NDTypeInterface>().getElementType();
                auto inActivationShape = getShape(inActivations);
                const auto dimY =
                        checked_cast_reg<NPUReg40XX::RegField_tensor_size_yType>(inActivationShape[Dims4D::Act::H]);
                const auto dimX =
                        checked_cast_reg<NPUReg40XX::RegField_tensor_size_xType>(inActivationShape[Dims4D::Act::W]);
                const auto dimZ =
                        checked_cast_reg<NPUReg40XX::RegField_tensor_size_zType>(inActivationShape[Dims4D::Act::C]);
                auto tensorMode = checked_cast_reg<NPUReg40XX::RegField_amodeType>(
                        getTensorMode<NPUReg40XX::RegField_amodeType>(inActivationsType));
                auto actDense = checked_cast_reg<NPUReg40XX::RegField_act_denseType>(!op.getInSparse());

                VPURegMapped::updateRegMappedInitializationValues(
                        initValues, {{"tensor_size0", {{"tensor_size_x", dimX}, {"tensor_size_y", dimY}}},
                                     {"tensor_size1", {{"tensor_size_z", dimZ}}},
                                     {"tensor_mode", {{"amode", tensorMode}}},
                                     {"kernel_pad_cfg", {{"act_dense", actDense}}}});
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUInputLayerCfgOp>(&IDUOp)) {
                auto sparsityPattern =
                        checked_cast_reg<NPUReg40XX::RegField_cm_sp_patternType>(op.getSparsityPattern());
                auto inputCompressed =
                        checked_cast_reg<NPUReg40XX::RegField_layer1_cmp_enType>(op.getInputCompressed());
                VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                                  {{"z_config", {{"cm_sp_pattern", sparsityPattern}}},
                                                                   {"kernel_pad_cfg",
                                                                    {{"act_dense", 1},
                                                                     {"wt_dense", 1},
                                                                     {"layer1_wt_sp_ins", 1},
                                                                     {"layer1_cmp_en", inputCompressed}}},
                                                                   {"tensor_size1", {{"tensor_size_z", 16}}}});
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUWeightsOp>(&IDUOp)) {
                auto wmode = checked_cast_reg<NPUReg40XX::RegField_wmodeType>(
                        getTensorMode<NPUReg40XX::RegField_wmodeType>(op.getWmode()));
                auto wtDense = checked_cast_reg<NPUReg40XX::RegField_wt_denseType>(!op.getWtSparse());
                VPURegMapped::updateRegMappedInitializationValues(
                        initValues, {{"tensor_mode", {{"wmode", wmode}}}, {"kernel_pad_cfg", {{"wt_dense", wtDense}}}});
                if (op.getPoolWtData().has_value()) {
                    auto poolWtData =
                            checked_cast_reg<NPUReg40XX::RegField_pool_wt_dataType>(op.getPoolWtData().value());
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues, {{"elops_wload", {{"pool_wt_data", poolWtData}}}});
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUWorkloadCfgOp>(&IDUOp)) {
                auto workloadType = op.getWorkloadType();
                switch (workloadType) {
                case VPUIPDPU::IDUWorkloadType::MAXPOOL:
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues,
                            {{"tensor_mode", {{"workload_operation", 0b10}, {"zm_input", 0b1}, {"dw_input", 0b1}}},
                             {"elops_wload", {{"pool_wt_rd_dis", 0b1}}},
                             {"kernel_pad_cfg", {{"dw_wt_sp_ins", 0b1}}}});
                    break;
                case VPUIPDPU::IDUWorkloadType::AVEPOOL:
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues,
                            {{"tensor_mode",
                              {{"workload_operation", 0}, {"zm_input", 0b1}, {"dw_input", 0b1}}},  // CONV workload
                             {"elops_wload", {{"pool_wt_rd_dis", 0b1}}},
                             {"kernel_pad_cfg", {{"dw_wt_sp_ins", 0b1}, {"dynamic_bw_en", 0b1}}}});

                    break;
                case VPUIPDPU::IDUWorkloadType::CONV:
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues,
                            {{"tensor_mode", {{"workload_operation", 0}, {"zm_input", 0b1}}}});  // CONV workload
                    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                                      {{"kernel_pad_cfg", {{"dynamic_bw_en", 1}}}});
                    break;
                case VPUIPDPU::IDUWorkloadType::DWCONV:
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues,
                            {{"tensor_mode",
                              {{"workload_operation", 0}, {"zm_input", 0b1}, {"dw_input", 0b1}}},  // CONV workload
                             {"kernel_pad_cfg", {{"dw_wt_sp_ins", 0b1}, {"dynamic_bw_en", 0b1}}}});
                    break;
                case VPUIPDPU::IDUWorkloadType::ELTWISE:
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues,
                            {{"tensor_mode", {{"workload_operation", 0}, {"zm_input", 0b1}}},  // CONV workload
                             {"kernel_pad_cfg", {{"dynamic_bw_en", 0b1}}},
                             {"elops_wload", {{"elop_wload", 0b1}}}});
                    break;
                default:
                    VPUX_THROW("Invalid Wload Type");
                    break;
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUDepthWiseCfgOp>(&IDUOp)) {
                if (op.getDw_3x3s1OptDis()) {
                    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                                      {{"base_offset_b", {{"dw_3x3s1_opt_dis", 1}}}});
                }
                if (op.getDwOptOffset().has_value()) {
                    auto dwOptOffset =
                            checked_cast_reg<NPUReg40XX::RegField_dw_opt_offsetType>(op.getDwOptOffset().value());
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues, {{"base_offset_b", {{"dw_opt_en", 1}, {"dw_opt_offset", dwOptOffset}}},
                                         {"kernel_pad_cfg", {{"pool_opt_en", 1}}}});
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUEltWiseCfgOp>(&IDUOp)) {
                auto elopScaleA = checked_cast_reg<NPUReg40XX::RegField_elop_scale_aType>(
                        op.getElopScaleAAttr().dyn_cast<mlir::IntegerAttr>().getInt());
                auto elopScaleB = checked_cast_reg<NPUReg40XX::RegField_elop_scale_bType>(
                        op.getElopScaleBAttr().dyn_cast<mlir::IntegerAttr>().getInt());
                VPURegMapped::updateRegMappedInitializationValues(
                        initValues, {{"elop_scale", {{"elop_scale_a", elopScaleA}, {"elop_scale_b", elopScaleB}}}});
            } else {
                VPUX_THROW("Unknown IDU operation: {0}", IDUOp);
            }
        }
    }
}

void DPUInvariantRewriter::fillMPECfg(mlir::Region& DPURegion,
                                      std::map<std::string, std::map<std::string, uint64_t>>& initValues) const {
    auto MPECfgOps = DPURegion.getOps<VPUIPDPU::MPECfgOp>();
    if (!MPECfgOps.empty()) {
        auto MPECfgOp = *MPECfgOps.begin();

        for (auto& MPEOp : MPECfgOp.getRegion().getOps()) {
            if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::MPEDenormalOperandsFTZOp>(&MPEOp)) {
                VPURegMapped::updateRegMappedInitializationValues(initValues, {{"mpe_cfg", {{"mpe_daz", 1}}}});
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::MPEActivationBiasOp>(&MPEOp)) {
                auto actBias = checked_cast_reg<NPUReg40XX::RegField_mpe_actbiasType>(op.getActBias());
                VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                                  {{"mpe_cfg", {{"mpe_actbias", actBias}}}});
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::MPEWeightsBiasOp>(&MPEOp)) {
                auto wtBias = checked_cast_reg<NPUReg40XX::RegField_mpe_wtbiasType>(op.getWeightsBias());
                VPURegMapped::updateRegMappedInitializationValues(initValues, {{"mpe_cfg", {{"mpe_wtbias", wtBias}}}});
            } else {
                VPUX_THROW("Unknown MPE operation: {0}", MPEOp);
            }
        }
    }
}

void DPUInvariantRewriter::fillPPECfg(mlir::Region& DPURegion,
                                      std::map<std::string, std::map<std::string, uint64_t>>& initValues) const {
    auto PPECgfOps = DPURegion.getOps<VPUIPDPU::PPECfgOp>();
    if (!PPECgfOps.empty()) {
        auto PPECgfOp = *PPECgfOps.begin();

        for (auto& PPEOp : PPECgfOp.getRegion().getOps()) {
            if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpBiasAddOp>(&PPEOp)) {
                VPUX_THROW_UNLESS((op.getScaleTable() != nullptr) ^ op.getBiasStatic().has_value(),
                                  "op {0} has ambiguous parameters", op);
                if (op.getScaleTable() != nullptr) {
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues, {
                                                {"ppe_scale_ctrl", {{"ppe_fp_scale_override", 0}}},
                                        });
                }
                if (op.getBiasStatic().has_value()) {
                    uint64_t biasStatic = checked_cast_reg<NPUReg40XX::RegField_ppe_fp_biasType>(
                            op.getBiasStatic().value().convertToFloat());
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues, {{"ppe_scale_ctrl", {{"ppe_fp_scale_override", 1}}},
                                         {"ppe_fp_bias", {{"ppe_fp_bias", biasStatic}}}});
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpScalePreluMultOp>(&PPEOp)) {
                VPUX_THROW_UNLESS((op.getScaleTable() != nullptr) ^ op.getScaleStatic().has_value(),
                                  "op {0} has ambiguous parameters", op);
                if (op.getScaleTable()) {
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues, {
                                                {"ppe_scale_ctrl", {{"ppe_fp_scale_override", 0}}},
                                        });
                }
                if (op.getScaleStatic().has_value()) {
                    uint64_t scaleStatic = checked_cast_reg<NPUReg40XX::RegField_ppe_fp_scaleType>(
                            op.getScaleStatic().value().convertToFloat());
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues, {{"ppe_scale_ctrl", {{"ppe_fp_scale_override", 1}}},
                                         {"ppe_fp_scale", {{"ppe_fp_scale", scaleStatic}}}});
                }
                if (op.getPreluAlpha().has_value()) {
                    uint64_t preluAlpha = checked_cast_reg<NPUReg40XX::RegField_ppe_fp_preluType>(
                            op.getPreluAlpha().value().convertToFloat());
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues, {{"ppe_fp_cfg", {{"ppe_fp_prelu_en", 1}}},
                                         {"ppe_fp_prelu", {{"ppe_fp_prelu", preluAlpha}}}});
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpAddMultBypassOp>(&PPEOp)) {
                auto bypassMode = checked_cast_reg<NPUReg40XX::RegField_ppe_fp_bypassType>(op.getBypassMode());
                VPURegMapped::updateRegMappedInitializationValues(
                        initValues, {
                                            {"ppe_fp_cfg", {{"ppe_fp_bypass", bypassMode}}},
                                    });
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpConvertOp>(&PPEOp)) {
                auto convertMode = checked_cast_reg<NPUReg40XX::RegField_ppe_fp_convertType>(op.getConvertMode());
                VPURegMapped::updateRegMappedInitializationValues(
                        initValues, {
                                            {"ppe_fp_cfg", {{"ppe_fp_convert", convertMode}}},
                                    });
                if (op.getClampMode().has_value()) {
                    auto clampMode =
                            checked_cast_reg<NPUReg40XX::RegField_ppe_fp16_clampType>(op.getClampMode().value());
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues, {
                                                {"ppe_misc", {{"ppe_fp16_clamp", clampMode}}},
                                        });
                }
                if (op.getFtzMode().has_value()) {
                    auto ftzMode = checked_cast_reg<NPUReg40XX::RegField_ppe_fp16_ftzType>(op.getFtzMode().value());
                    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                                      {
                                                                              {"ppe_misc", {{"ppe_fp16_ftz", ftzMode}}},
                                                                      });
                }
                if (op.getBf16RoundMode().has_value()) {
                    auto bf16RoundMode =
                            checked_cast_reg<NPUReg40XX::RegField_ppe_bf16_roundType>(op.getBf16RoundMode().value());
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues, {
                                                {"ppe_fp_cfg", {{"ppe_bf16_round", bf16RoundMode}}},
                                        });
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntBiasAddOp>(&PPEOp)) {
                VPUX_THROW_UNLESS((op.getScaleTable() != nullptr) ^ op.getBiasStatic().has_value(),
                                  "op {0} has ambiguous parameters", op);
                if (op.getScaleTable() != nullptr) {
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues, {
                                                {"ppe_scale_ctrl", {{"ppe_scale_override", 0}}},
                                        });
                }
                if (op.getBiasStatic().has_value()) {
                    uint64_t biasStatic =
                            checked_cast_reg<NPUReg40XX::RegField_ppe_biasType>(op.getBiasStatic().value());
                    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                                      {{"ppe_scale_ctrl", {{"ppe_scale_override", 1}}},
                                                                       {"ppe_bias", {{"ppe_bias", biasStatic}}}});
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntScaleMultOp>(&PPEOp)) {
                VPUX_THROW_UNLESS((op.getScaleTable() != nullptr) ^ op.getScaleStatic().has_value(),
                                  "op {0} has ambiguous parameters", op);
                if (op.getScaleTable()) {
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues, {
                                                {"ppe_scale_ctrl", {{"ppe_scale_override", 0}}},
                                        });
                }
                if (op.getScaleStatic().has_value()) {
                    uint64_t scaleStatic =
                            checked_cast_reg<NPUReg40XX::RegField_ppe_scale_multType>(op.getScaleStatic().value());
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues, {{"ppe_scale_ctrl", {{"ppe_scale_override", 1}}},
                                         {"ppe_scale", {{"ppe_scale_mult", scaleStatic}}}});
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntPreluMultOp>(&PPEOp)) {
                auto preluMultStatic =
                        checked_cast_reg<NPUReg40XX::RegField_ppe_prelu_multType>(op.getPreluMultStatic());
                VPURegMapped::updateRegMappedInitializationValues(
                        initValues, {
                                            {"ppe_prelu", {{"ppe_prelu_mult", preluMultStatic}}},
                                    });
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntScaleShiftOp>(&PPEOp)) {
                VPUX_THROW_UNLESS((op.getScaleTable() != nullptr) ^ op.getShiftStatic().has_value(),
                                  "op {0} has ambiguous parameters", op);
                if (op.getScaleTable() != nullptr) {
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues, {
                                                {"ppe_scale_ctrl", {{"ppe_scale_override", 0}}},
                                        });
                }
                if (op.getShiftStatic().has_value()) {
                    uint64_t shiftStatic =
                            checked_cast_reg<NPUReg40XX::RegField_ppe_scale_shiftType>(op.getShiftStatic().value());
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues, {{"ppe_scale_ctrl", {{"ppe_scale_override", 1}}},
                                         {"ppe_scale", {{"ppe_scale_shift", shiftStatic}}}});
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntPreluShiftOp>(&PPEOp)) {
                auto preluShiftStatic =
                        checked_cast_reg<NPUReg40XX::RegField_ppe_prelu_shiftType>(op.getPreluShiftStatic());
                VPURegMapped::updateRegMappedInitializationValues(
                        initValues, {
                                            {"ppe_prelu", {{"ppe_prelu_shift", preluShiftStatic}}},
                                    });
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntRoundOp>(&PPEOp)) {
                auto roundMode = checked_cast_reg<NPUReg40XX::RegField_ppe_scale_roundType>(op.getRoundMode());
                VPURegMapped::updateRegMappedInitializationValues(
                        initValues, {
                                            {"ppe_scale", {{"ppe_scale_round", roundMode}}},
                                    });
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntZeroPointOffsetOp>(&PPEOp)) {
                auto zeroPointStatic =
                        checked_cast_reg<NPUReg40XX::RegField_ppe_g8_bias_cType>(op.getZeroPointStatic());
                VPURegMapped::updateRegMappedInitializationValues(
                        initValues, {
                                            {"ppe_cfg", {{"ppe_g8_bias_c", zeroPointStatic}}},
                                    });
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntClampOp>(&PPEOp)) {
                auto clampHigh = checked_cast_reg<NPUReg40XX::RegField_ppe_scale_hclampType>(op.getClampHigh());
                VPURegMapped::updateRegMappedInitializationValues(
                        initValues, {
                                            {"ppe_scale_hclamp", {{"ppe_scale_hclamp", clampHigh}}},
                                    });
                if (op.getClampLow().has_value()) {
                    auto clampLow =
                            checked_cast_reg<NPUReg40XX::RegField_ppe_scale_lclampType>(op.getClampLow().value());
                    VPURegMapped::updateRegMappedInitializationValues(
                            initValues, {
                                                {"ppe_scale_lclamp", {{"ppe_scale_lclamp", clampLow}}},
                                        });
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntConvertOp>(&PPEOp)) {
                auto convertMode = checked_cast_reg<NPUReg40XX::RegField_ppe_i32_convertType>(op.getConvertMode());
                VPURegMapped::updateRegMappedInitializationValues(
                        initValues, {
                                            {"ppe_misc", {{"ppe_i32_convert", convertMode}}},
                                    });
            } else {
                VPUX_THROW("Unknown PPE operation {0}", PPEOp);
            }
        }
    }
}

void DPUInvariantRewriter::fillODUCfg(mlir::Region& DPURegion,
                                      std::map<std::string, std::map<std::string, uint64_t>>& initValues) const {
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
            if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUOutTensorSizeOp>(&ODUOp)) {
                auto xDim = checked_cast_reg<NPUReg40XX::RegField_te_dim_xType>(op.getDimX() - 1);
                auto yDim = checked_cast_reg<NPUReg40XX::RegField_te_dim_yType>(op.getDimY() - 1);
                auto zDim = checked_cast_reg<NPUReg40XX::RegField_te_dim_zType>(op.getDimZ() - 1);
                VPURegMapped::updateRegMappedInitializationValues(
                        initValues, {
                                            {"te_dim0", {{"te_dim_y", yDim}, {"te_dim_z", zDim}}},
                                            {"te_dim1", {{"te_dim_x", xDim}}},
                                    });
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUDataReuseOp>(&ODUOp)) {
                auto activationReuse = checked_cast_reg<NPUReg40XX::RegField_nthwType>(op.getActivationReuse());
                VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                                  {
                                                                          {"odu_cfg", {{"nthw", activationReuse}}},
                                                                  });
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUPermuteDataOp>(&ODUOp)) {
                auto permuteMode = checked_cast_reg<NPUReg40XX::RegField_permutationType>(op.getPermuteMode());
                VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                                  {
                                                                          {"odu_cfg", {{"permutation", permuteMode}}},
                                                                  });
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUSparsityOp>(&ODUOp)) {
                auto sparseValue = checked_cast_reg<NPUReg40XX::RegField_sp_valueType>(op.getSparseValue().value_or(0));
                auto compressionEnabled =
                        checked_cast_reg<NPUReg40XX::RegField_sp_out_enType>(op.getCompressionEnabled().value_or(true));
                uint64_t writeSp = op.getSparsityMap() ? 1 : 0;
                VPURegMapped::updateRegMappedInitializationValues(
                        initValues,
                        {
                                {"odu_cfg",
                                 {{"sp_value", sparseValue}, {"sp_out_en", compressionEnabled}, {"write_sp", writeSp}}},
                        });
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUSwizzleDataOp>(&ODUOp)) {
                auto swizzleKey = checked_cast_reg<NPUReg40XX::RegField_swizzle_keyType>(op.getSwizzleKey());
                VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                                  {
                                                                          {"odu_cfg", {{"swizzle_key", swizzleKey}}},
                                                                  });
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUOutActivationsOp>(&ODUOp)) {
                uint64_t dataWidth(0);
                if (op.getDataWidth().has_value()) {
                    dataWidth = checked_cast_reg<NPUReg40XX::RegField_dtypeType>(op.getDataWidth().value());
                } else {
                    auto outActType = op.getOutActivations().getType().cast<mlir::MemRefType>().getElementType();
                    dataWidth = checked_cast_reg<NPUReg40XX::RegField_dtypeType>(getDataBitWidth(outActType));
                }

                VPURegMapped::updateRegMappedInitializationValues(
                        initValues, {
                                            {"odu_cfg", {{"dtype", dataWidth}, {"write_ac", 1}}},
                                    });
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUMemoryModeOp>(&ODUOp)) {
                auto memMode = checked_cast_reg<NPUReg40XX::RegField_modeType>(op.getMemMode());
                VPURegMapped::updateRegMappedInitializationValues(initValues, {
                                                                                      {"odu_cfg", {{"mode", memMode}}},
                                                                              });
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUCmxPortsOp>(&ODUOp)) {
                auto cmxPorts = checked_cast_reg<NPUReg40XX::RegField_cmx_port_muxing_disableType>(op.getCmxPorts());
                VPURegMapped::updateRegMappedInitializationValues(
                        initValues, {
                                            {"odu_cfg", {{"cmx_port_muxing_disable", cmxPorts}}},
                                    });
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUWriteCombineBufferOp>(&ODUOp)) {
                auto activationsMode = checked_cast_reg<NPUReg40XX::RegField_wcb_ac_modeType>(op.getActivationsMode());
                auto sparsityMode = checked_cast_reg<NPUReg40XX::RegField_wcb_sp_modeType>(
                        op.getSparsityMode().value_or(vpux::VPUIPDPU::ODUWcbCombineMode::WCB_COMBINE_BY_CONTEXT));

                VPURegMapped::updateRegMappedInitializationValues(
                        initValues,
                        {
                                {"odu_cfg",
                                 {{"wcb_bypass", 0}, {"wcb_ac_mode", activationsMode}, {"wcb_sp_mode", sparsityMode}}},
                        });
            } else {
                VPUX_THROW("Unknown ODU operation {0}", ODUOp);
            }
        }
    }
}

void DPUInvariantRewriter::fillBarrierCfg(VPUIPDPU::DPUInvariantOp origOp,
                                          std::map<std::string, std::map<std::string, uint64_t>>& initValues) const {
    auto barrierCfgOps = to_small_vector(origOp.getRegion().getOps<VPUIPDPU::BarrierCfgOp>());
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

    return;
}

void DPUInvariantRewriter::fillProfilingCfg(VPUIPDPU::DPUInvariantOp origOp,
                                            std::map<std::string, std::map<std::string, uint64_t>>& initValues) const {
    if (!origOp.getProfilingData().has_value()) {
        return;
    }
    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                      {{"hwp_ctrl", {{"hwp_en", 1}, {"hwp_stat_mode", 3}}}});
}

void DPUInvariantRewriter::fillStubCfg(std::map<std::string, std::map<std::string, uint64_t>>& initValues) const {
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
