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
using namespace NPUReg40XX;

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
        } else if (type.isFloat8E5M2()) {
            return static_cast<uint64_t>(nn_public::VpuInputTensorDType::FP8);
        } else if (type.isFloat8E4M3FN()) {
            return static_cast<uint64_t>(nn_public::VpuInputTensorDType::RESERVED);
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

    vpux::NPUReg40XX::Descriptors::DpuInvariantRegister descriptor;
    // fill default configuration
    descriptor.write<Fields::cmx_slice0_low_addr>(0x4000000);
    descriptor.write<Fields::cmx_slice1_low_addr>(0x4000000);
    descriptor.write<Fields::cmx_slice2_low_addr>(0x4000000);
    descriptor.write<Fields::cmx_slice3_low_addr>(0x4000000);
    descriptor.write<Fields::cmx_slice_size>(0x00018000);
    descriptor.write<Fields::ppe_scale_round>(3);
    descriptor.write<Fields::ppe_prelu_mult>(0x1);
    descriptor.write<Fields::ppe_scale_hclamp>(0x7FFFFFFF);
    descriptor.write<Fields::ppe_scale_lclamp>(int64_t(-2147483648));  // 0x80000000

    if (_dryRunMode == VPU::DPUDryRunMode::STUB) {
        _log.trace("DPU dry run mode = 'stub', updating invariant descriptor");
        fillStubCfg(descriptor);
    } else {
        fillIDUCfg(origOp.getRegion(), descriptor);
        fillMPECfg(origOp.getRegion(), descriptor);
        fillPPECfg(origOp.getRegion(), descriptor);
        fillODUCfg(origOp.getRegion(), descriptor);
    }
    fillBarrierCfg(origOp, descriptor);
    fillProfilingCfg(origOp, descriptor);

    auto taskListCfgOp = to_small_vector(origOp.getRegion().getOps<VPUIPDPU::DPUGroupOp>());
    if (taskListCfgOp.size() == 1) {
        descriptor.write<Fields::variant_count_>(taskListCfgOp[0].getVariantCount());
    }

    descriptor.write<Fields::nvar_tag>(origOp.getIndex() + 1);

    auto regDPUInvariantAttr = DpuInvariantRegisterAttr::get(rewriter.getContext(), std::move(descriptor));

    rewriter.create<NPUReg40XX::DPUInvariantOp>(
            origOp->getLoc(), origOp.getSymNameAttr(), origOp.getTaskIndexAttr(), regDPUInvariantAttr,
            origOp.getTaskLocationAttr(), origOp.getInputAttr(), origOp.getInputSparsityMapAttr(),
            origOp.getInputStorageElementTableAttr(), origOp.getWeightsAttr(), origOp.getWeightsSparsityMapAttr(),
            origOp.getWeightTableAttr(), origOp.getSprLookupTableAttr(), origOp.getOutputAttr(),
            origOp.getOutputSparsityMapAttr(), origOp.getProfilingDataAttr(), origOp.getIsZeroOffsetWeightsTableAttr(),
            origOp.getNceTaskTypeAttr(), origOp.getIsContinuedAttr());

    rewriter.eraseOp(origOp);

    return mlir::success();
}

void DPUInvariantRewriter::fillIDUCfg(mlir::Region& DPURegion,
                                      vpux::NPUReg40XX::Descriptors::DpuInvariantRegister& descriptor) const {
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
                    descriptor.write<Fields::se_z_split>(seSizeHW);
                } else {
                    auto seSizeHW = ((seSize + 15) >> 4) - 1;
                    descriptor.write<Fields::npo2_se_z_split_en>(1);
                    descriptor.write<Fields::npo2_se_size>(seSizeHW);
                }

                if (op.getNumSesInZDir().has_value()) {
                    descriptor.write<Fields::num_ses_in_z_dir>(op.getNumSesInZDir().value());
                }

            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUKernelOp>(&IDUOp)) {
                descriptor.write<Fields::kernel_y>(op.getKernelY());
                descriptor.write<Fields::kernel_x>(op.getKernelX());
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUStrideOp>(&IDUOp)) {
                auto strideY = op.getStrideY() - 1;
                auto strideX = op.getStrideX() - 1;
                descriptor.write<Fields::stride>(strideX);
                if (op.getStrideY() != op.getStrideX()) {
                    descriptor.write<Fields::stride_y>(strideY);
                    descriptor.write<Fields::stride_y_en>(1);
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUInActivationsOp>(&IDUOp)) {
                auto inActivations = op.getInActivations();
                auto inActivationsType = inActivations.getType().cast<vpux::NDTypeInterface>().getElementType();
                auto inActivationShape = getShape(inActivations);
                const auto dimY = inActivationShape[Dims4D::Act::H];
                const auto dimX = inActivationShape[Dims4D::Act::W];
                const auto dimZ = inActivationShape[Dims4D::Act::C];
                auto tensorMode = getTensorMode<NPUReg40XX::RegField_amodeType>(inActivationsType);
                auto actDense = !op.getInSparse();

                descriptor.write<Fields::tensor_size_x>(dimX);
                descriptor.write<Fields::tensor_size_y>(dimY);
                descriptor.write<Fields::tensor_size_z>(dimZ);
                descriptor.write<Fields::amode>(tensorMode);
                descriptor.write<Fields::act_dense>(actDense);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUInputLayerCfgOp>(&IDUOp)) {
                descriptor.write<Fields::cm_sp_pattern>(op.getSparsityPattern());
                descriptor.write<Fields::act_dense>(1);
                descriptor.write<Fields::wt_dense>(1);
                descriptor.write<Fields::layer1_wt_sp_ins>(1);
                descriptor.write<Fields::layer1_cmp_en>(op.getInputCompressed());
                descriptor.write<Fields::tensor_size_z>(16);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUWeightsOp>(&IDUOp)) {
                auto wmode = getTensorMode<NPUReg40XX::RegField_wmodeType>(op.getWmode());
                auto wtDense = !op.getWtSparse();
                descriptor.write<Fields::wmode>(wmode);
                descriptor.write<Fields::wt_dense>(wtDense);
                descriptor.write<Fields::wt_plt_cfg>(op.getWtPltCfg());
                if (op.getPoolWtData().has_value()) {
                    descriptor.write<Fields::pool_wt_data>(op.getPoolWtData().value());
                }
                if (op.getQuantilesLut().has_value()) {
                    auto quantilesLut = op.getQuantilesLut().value();
                    constexpr unsigned numPalletTableEntries = 16;
                    VPUX_THROW_UNLESS((quantilesLut.size() <= numPalletTableEntries),
                                      "Number of palletization table entries ({0}) exceeds maximum of 16",
                                      quantilesLut.size());
                    llvm::SmallVector<uint16_t, numPalletTableEntries> quantilesLutValues(numPalletTableEntries, 0);

                    auto getPalletModeBitValue = [](const double value, const uint64_t wmode) -> uint16_t {
                        if (wmode == static_cast<uint64_t>(nn_public::VpuInputTensorDType::FP16)) {
                            vpux::type::float16 f16(value);
                            return f16.to_bits();
                        } else if (wmode == static_cast<uint64_t>(nn_public::VpuInputTensorDType::U8)) {
                            int i8 = static_cast<int>(value);
                            return (i8 < 0 ? 0 : static_cast<uint16_t>(i8));
                        } else if (wmode == static_cast<uint64_t>(nn_public::VpuInputTensorDType::I8)) {
                            return static_cast<uint16_t>(static_cast<int>(value));
                        } else if (wmode == static_cast<uint64_t>(nn_public::VpuInputTensorDType::BF16)) {
                            vpux::type::bfloat16 bf16(value);
                            return bf16.to_bits();
                        } else if (wmode == static_cast<uint64_t>(nn_public::VpuInputTensorDType::FP8)) {
                            vpux::type::float8_e5m2 bf8(value);
                            return bf8.to_bits();
                        } else if (wmode == static_cast<uint64_t>(nn_public::VpuInputTensorDType::RESERVED)) {
                            vpux::type::float8_e4m3 hf8(value);
                            return hf8.to_bits();
                        } else {
                            VPUX_THROW("getPalletModeBitValue: Unsupported wmode for palletization table {0}", wmode);
                        }
                        return 0;
                    };

                    for (unsigned i = 0; i < quantilesLut.size(); ++i) {
                        double lutEntry = quantilesLut[i].dyn_cast<mlir::FloatAttr>().getValueAsDouble();
                        quantilesLutValues[i] = getPalletModeBitValue(lutEntry, wmode);
                    }
                    descriptor.write<Fields::plt_idx_0>(quantilesLutValues[0]);
                    descriptor.write<Fields::plt_idx_1>(quantilesLutValues[1]);
                    descriptor.write<Fields::plt_idx_2>(quantilesLutValues[2]);
                    descriptor.write<Fields::plt_idx_3>(quantilesLutValues[3]);
                    descriptor.write<Fields::plt_idx_4>(quantilesLutValues[4]);
                    descriptor.write<Fields::plt_idx_5>(quantilesLutValues[5]);
                    descriptor.write<Fields::plt_idx_6>(quantilesLutValues[6]);
                    descriptor.write<Fields::plt_idx_7>(quantilesLutValues[7]);
                    descriptor.write<Fields::plt_idx_8>(quantilesLutValues[8]);
                    descriptor.write<Fields::plt_idx_9>(quantilesLutValues[9]);
                    descriptor.write<Fields::plt_idx_10>(quantilesLutValues[10]);
                    descriptor.write<Fields::plt_idx_11>(quantilesLutValues[11]);
                    descriptor.write<Fields::plt_idx_12>(quantilesLutValues[12]);
                    descriptor.write<Fields::plt_idx_13>(quantilesLutValues[13]);
                    descriptor.write<Fields::plt_idx_14>(quantilesLutValues[14]);
                    descriptor.write<Fields::plt_idx_15>(quantilesLutValues[15]);
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUWorkloadCfgOp>(&IDUOp)) {
                auto workloadType = op.getWorkloadType();
                switch (workloadType) {
                case VPUIPDPU::IDUWorkloadType::MAXPOOL:
                    descriptor.write<Fields::workload_operation>(0b10);
                    descriptor.write<Fields::zm_input>(0b1);
                    descriptor.write<Fields::dw_input>(0b1);
                    descriptor.write<Fields::pool_wt_rd_dis>(0b1);
                    descriptor.write<Fields::dw_wt_sp_ins>(0b1);
                    break;
                case VPUIPDPU::IDUWorkloadType::AVEPOOL:
                    descriptor.write<Fields::workload_operation>(0);
                    descriptor.write<Fields::zm_input>(0b1);  // CONV workload
                    descriptor.write<Fields::dw_input>(0b1);
                    descriptor.write<Fields::pool_wt_rd_dis>(0b1);
                    descriptor.write<Fields::dw_wt_sp_ins>(0b1);
                    descriptor.write<Fields::dynamic_bw_en>(0b1);
                    break;
                case VPUIPDPU::IDUWorkloadType::CONV:
                    descriptor.write<Fields::workload_operation>(0);
                    descriptor.write<Fields::zm_input>(0b1);  // CONV workload
                    descriptor.write<Fields::dynamic_bw_en>(1);
                    break;
                case VPUIPDPU::IDUWorkloadType::DWCONV:
                    descriptor.write<Fields::workload_operation>(0);
                    descriptor.write<Fields::zm_input>(0b1);  // CONV workload
                    descriptor.write<Fields::dw_input>(0b1);
                    descriptor.write<Fields::dw_wt_sp_ins>(0b1);
                    descriptor.write<Fields::dynamic_bw_en>(0b1);
                    break;
                case VPUIPDPU::IDUWorkloadType::ELTWISE:
                    descriptor.write<Fields::workload_operation>(0);
                    descriptor.write<Fields::zm_input>(0b1);  // CONV workload
                    descriptor.write<Fields::dynamic_bw_en>(0b1);
                    descriptor.write<Fields::elop_wload>(0b1);
                    break;
                default:
                    VPUX_THROW("Invalid Wload Type");
                    break;
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUDepthWiseCfgOp>(&IDUOp)) {
                if (op.getDw_3x3s1OptDis()) {
                    descriptor.write<Fields::dw_3x3s1_opt_dis>(1);
                }
                if (op.getDwOptOffset().has_value()) {
                    descriptor.write<Fields::dw_opt_en>(1);
                    descriptor.write<Fields::dw_opt_offset>(op.getDwOptOffset().value());
                    descriptor.write<Fields::pool_opt_en>(1);
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::IDUEltWiseCfgOp>(&IDUOp)) {
                auto elopScaleA = op.getElopScaleAAttr().dyn_cast<mlir::IntegerAttr>().getInt();
                auto elopScaleB = op.getElopScaleBAttr().dyn_cast<mlir::IntegerAttr>().getInt();
                descriptor.write<Fields::elop_scale_a>(elopScaleA);
                descriptor.write<Fields::elop_scale_b>(elopScaleB);
            } else {
                VPUX_THROW("Unknown IDU operation: {0}", IDUOp);
            }
        }
    }
}

void DPUInvariantRewriter::fillMPECfg(mlir::Region& DPURegion,
                                      vpux::NPUReg40XX::Descriptors::DpuInvariantRegister& descriptor) const {
    auto MPECfgOps = DPURegion.getOps<VPUIPDPU::MPECfgOp>();
    if (!MPECfgOps.empty()) {
        auto MPECfgOp = *MPECfgOps.begin();

        for (auto& MPEOp : MPECfgOp.getRegion().getOps()) {
            if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::MPEDenormalOperandsFTZOp>(&MPEOp)) {
                descriptor.write<Fields::mpe_daz>(1);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::MPEActivationBiasOp>(&MPEOp)) {
                descriptor.write<Fields::mpe_actbias>(op.getActBias());
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::MPEWeightsBiasOp>(&MPEOp)) {
                descriptor.write<Fields::mpe_wtbias>(op.getWeightsBias());
            } else {
                VPUX_THROW("Unknown MPE operation: {0}", MPEOp);
            }
        }
    }
}

void DPUInvariantRewriter::fillPPECfg(mlir::Region& DPURegion,
                                      vpux::NPUReg40XX::Descriptors::DpuInvariantRegister& descriptor) const {
    auto PPECgfOps = DPURegion.getOps<VPUIPDPU::PPECfgOp>();
    if (!PPECgfOps.empty()) {
        auto PPECgfOp = *PPECgfOps.begin();

        for (auto& PPEOp : PPECgfOp.getRegion().getOps()) {
            if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpBiasAddOp>(&PPEOp)) {
                VPUX_THROW_UNLESS((op.getScaleTable() != nullptr) ^ op.getBiasStatic().has_value(),
                                  "op {0} has ambiguous parameters", op);
                if (op.getScaleTable() != nullptr) {
                    descriptor.write<Fields::ppe_fp_scale_override>(0);
                }
                if (op.getBiasStatic().has_value()) {
                    auto biasStatic = op.getBiasStatic().value().convertToFloat();
                    descriptor.write<Fields::ppe_fp_scale_override>(1);
                    descriptor.write<Fields::ppe_fp_bias>(biasStatic);
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpScalePreluMultOp>(&PPEOp)) {
                VPUX_THROW_UNLESS((op.getScaleTable() != nullptr) ^ op.getScaleStatic().has_value(),
                                  "op {0} has ambiguous parameters", op);
                if (op.getScaleTable()) {
                    descriptor.write<Fields::ppe_fp_scale_override>(0);
                }
                if (op.getScaleStatic().has_value()) {
                    auto scaleStatic = op.getScaleStatic().value().convertToFloat();
                    descriptor.write<Fields::ppe_fp_scale_override>(1);
                    descriptor.write<Fields::ppe_fp_scale>(scaleStatic);
                }
                if (op.getPreluAlpha().has_value()) {
                    auto preluAlpha = op.getPreluAlpha().value().convertToFloat();
                    descriptor.write<Fields::ppe_fp_prelu_en>(1);
                    descriptor.write<Fields::ppe_fp_prelu>(preluAlpha);
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpAddMultBypassOp>(&PPEOp)) {
                descriptor.write<Fields::ppe_fp_bypass>(op.getBypassMode());
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEFpConvertOp>(&PPEOp)) {
                descriptor.write<Fields::ppe_fp_convert>(op.getConvertMode());
                if (op.getClampMode().has_value()) {
                    descriptor.write<Fields::ppe_fp16_clamp>(op.getClampMode().value());
                }
                if (op.getFtzMode().has_value()) {
                    descriptor.write<Fields::ppe_fp16_ftz>(op.getFtzMode().value());
                }
                if (op.getBf16RoundMode().has_value()) {
                    descriptor.write<Fields::ppe_bf16_round>(op.getBf16RoundMode().value());
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntBiasAddOp>(&PPEOp)) {
                VPUX_THROW_UNLESS((op.getScaleTable() != nullptr) ^ op.getBiasStatic().has_value(),
                                  "op {0} has ambiguous parameters", op);
                if (op.getScaleTable() != nullptr) {
                    descriptor.write<Fields::ppe_scale_override>(0);
                }
                if (op.getBiasStatic().has_value()) {
                    descriptor.write<Fields::ppe_scale_override>(1);
                    descriptor.write<Fields::ppe_bias>(op.getBiasStatic().value());
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntScaleMultOp>(&PPEOp)) {
                VPUX_THROW_UNLESS((op.getScaleTable() != nullptr) ^ op.getScaleStatic().has_value(),
                                  "op {0} has ambiguous parameters", op);
                if (op.getScaleTable()) {
                    descriptor.write<Fields::ppe_scale_override>(0);
                }
                if (op.getScaleStatic().has_value()) {
                    descriptor.write<Fields::ppe_scale_override>(1);
                    descriptor.write<Fields::ppe_scale_mult>(op.getScaleStatic().value());
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntPreluMultOp>(&PPEOp)) {
                descriptor.write<Fields::ppe_prelu_mult>(op.getPreluMultStatic());
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntScaleShiftOp>(&PPEOp)) {
                VPUX_THROW_UNLESS((op.getScaleTable() != nullptr) ^ op.getShiftStatic().has_value(),
                                  "op {0} has ambiguous parameters", op);
                if (op.getScaleTable() != nullptr) {
                    descriptor.write<Fields::ppe_scale_override>(0);
                }
                if (op.getShiftStatic().has_value()) {
                    descriptor.write<Fields::ppe_scale_override>(1);
                    descriptor.write<Fields::ppe_scale_shift>(op.getShiftStatic().value());
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntPreluShiftOp>(&PPEOp)) {
                descriptor.write<Fields::ppe_prelu_shift>(op.getPreluShiftStatic());
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntRoundOp>(&PPEOp)) {
                descriptor.write<Fields::ppe_scale_round>(op.getRoundMode());
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntZeroPointOffsetOp>(&PPEOp)) {
                descriptor.write<Fields::ppe_g8_bias_c>(op.getZeroPointStatic());
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntClampOp>(&PPEOp)) {
                descriptor.write<Fields::ppe_scale_hclamp>(op.getClampHigh());
                if (op.getClampLow().has_value()) {
                    descriptor.write<Fields::ppe_scale_lclamp>(op.getClampLow().value());
                }
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::PPEIntConvertOp>(&PPEOp)) {
                descriptor.write<Fields::ppe_i32_convert>(op.getConvertMode());
            } else {
                VPUX_THROW("Unknown PPE operation {0}", PPEOp);
            }
        }
    }
}

void DPUInvariantRewriter::fillODUCfg(mlir::Region& DPURegion,
                                      vpux::NPUReg40XX::Descriptors::DpuInvariantRegister& descriptor) const {
    auto ODUCfgOps = DPURegion.getOps<VPUIPDPU::ODUCfgOp>();
    if (!ODUCfgOps.empty()) {
        auto ODUCfgOp = *ODUCfgOps.begin();

        // TODO: E#80766 select optimal write combine mode and serialize based on VPUIPDU instruction
        descriptor.write<Fields::wcb_bypass>(0);

        // Statically set bits that should not be part of functional defaults

        // Not used by HW. Setting to 1 to be coeherent with GFile.
        descriptor.write<Fields::addr_format_sel>(1);

        // TODO: E#81883 need to figure this out why it's always set to 1?
        descriptor.write<Fields::rst_ctxt>(1);

        // TODO: E#82814 should it be a  defailt value? this is hardcoded and directly copied from POC runtime...
        descriptor.write<Fields::base_offset_a>(0x200);

        for (auto& ODUOp : ODUCfgOp.getRegion().getOps()) {
            if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUOutTensorSizeOp>(&ODUOp)) {
                auto xDim = op.getDimX() - 1;
                auto yDim = op.getDimY() - 1;
                auto zDim = op.getDimZ() - 1;
                descriptor.write<Fields::te_dim_y>(yDim);
                descriptor.write<Fields::te_dim_z>(zDim);
                descriptor.write<Fields::te_dim_x>(xDim);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUDataReuseOp>(&ODUOp)) {
                descriptor.write<Fields::nthw>(op.getActivationReuse());
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUPermuteDataOp>(&ODUOp)) {
                descriptor.write<Fields::permutation>(op.getPermuteMode());
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUSparsityOp>(&ODUOp)) {
                uint64_t writeSp = op.getSparsityMap() ? 1 : 0;
                descriptor.write<Fields::sp_value>(op.getSparseValue().value_or(0));
                descriptor.write<Fields::sp_out_en>(op.getCompressionEnabled().value_or(true));
                descriptor.write<Fields::write_sp>(writeSp);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUSwizzleDataOp>(&ODUOp)) {
                descriptor.write<Fields::swizzle_key>(op.getSwizzleKey());
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUOutActivationsOp>(&ODUOp)) {
                uint64_t dataWidth(0);
                if (op.getDataWidth().has_value()) {
                    dataWidth = static_cast<uint64_t>(op.getDataWidth().value());
                } else {
                    auto outActType = op.getOutActivations().getType().cast<mlir::MemRefType>().getElementType();
                    dataWidth = static_cast<uint64_t>(getDataBitWidth(outActType));
                }

                descriptor.write<Fields::dtype>(dataWidth);
                descriptor.write<Fields::write_ac>(1);
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUMemoryModeOp>(&ODUOp)) {
                descriptor.write<Fields::mode>(op.getMemMode());
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUCmxPortsOp>(&ODUOp)) {
                descriptor.write<Fields::cmx_port_muxing_disable>(op.getCmxPorts());
            } else if (auto op = mlir::dyn_cast_or_null<VPUIPDPU::ODUWriteCombineBufferOp>(&ODUOp)) {
                descriptor.write<Fields::wcb_bypass>(0);
                descriptor.write<Fields::wcb_ac_mode>(op.getActivationsMode());
                descriptor.write<Fields::wcb_sp_mode>(
                        op.getSparsityMode().value_or(vpux::VPUIPDPU::ODUWcbCombineMode::WCB_COMBINE_BY_CONTEXT));
            } else {
                VPUX_THROW("Unknown ODU operation {0}", ODUOp);
            }
        }
    }
}

void DPUInvariantRewriter::fillBarrierCfg(VPUIPDPU::DPUInvariantOp origOp,
                                          vpux::NPUReg40XX::Descriptors::DpuInvariantRegister& descriptor) const {
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

        descriptor.write<Fields::group_>(barrierGroup);
        descriptor.write<Fields::mask_>(barrierMask);
        descriptor.write<Registers::barriers_sched_, Fields::start_after_>(startAfter);
        descriptor.write<Registers::barriers_sched_, Fields::clean_after_>(cleanAfter);
        descriptor.write<Fields::barriers_wait_mask_hi_>(consMaskHi);
        descriptor.write<Fields::barriers_wait_mask_lo_>(consMaskLo);
        descriptor.write<Fields::barriers_post_mask_hi_>(prodMaskHi);
        descriptor.write<Fields::barriers_post_mask_lo_>(prodMaskLo);
    } else {
        // just to explicitly show that we really intentionally only care about size == 1
        return;
    }

    return;
}

void DPUInvariantRewriter::fillProfilingCfg(VPUIPDPU::DPUInvariantOp origOp,
                                            vpux::NPUReg40XX::Descriptors::DpuInvariantRegister& descriptor) const {
    if (!origOp.getProfilingData().has_value()) {
        return;
    }
    descriptor.write<Fields::hwp_en>(1);
    descriptor.write<Fields::hwp_stat_mode>(3);
}

void DPUInvariantRewriter::fillStubCfg(vpux::NPUReg40XX::Descriptors::DpuInvariantRegister& descriptor) const {
    descriptor.write<Fields::tensor_size_x>(0x1);
    descriptor.write<Fields::tensor_size_y>(0x1);
    descriptor.write<Fields::tensor_size_z>(0x10);
    descriptor.write<Fields::workload_operation>(0x0);
    descriptor.write<Fields::zm_input>(0x1);
    descriptor.write<Fields::kernel_y>(0x1);
    descriptor.write<Fields::kernel_x>(0x1);
    descriptor.write<Fields::elop_wload>(0x1);
    descriptor.write<Fields::elop_wload_type>(0x1);
    descriptor.write<Fields::te_dim_y>(0x0);
    descriptor.write<Fields::te_dim_z>(0xF);
    descriptor.write<Fields::te_dim_x>(0x0);
    descriptor.write<Fields::nthw>(0x1);
}

}  // namespace vpuipdpu2npureg40xx
}  // namespace vpux
