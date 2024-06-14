//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/utils.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"
#include "vpux/compiler/utils/compression_utils.hpp"
#include "vpux/compiler/utils/elf_utils.hpp"
#include "vpux/utils/core/mem_size.hpp"

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
        if (type.isSignedInteger() || type.isUnsignedInteger() || type.isSignlessInteger()) {
            return DMA_ACC_DTYPE_INT8_UINT8;
        } else if (type.isBF16() || type.isF16()) {
            return DMA_ACC_DTYPE_FP16_BF16;
        }
    }

    VPUX_THROW("Invalid tensor type for DMA Acceleration configuration {0}", type);
}

using RegisterMap = std::map<std::string, std::map<std::string, uint64_t>>;

void setDMAConversionMode(RegisterMap& initValues, mlir::Type inputType, uint64_t srcSize, mlir::Type outputType,
                          uint64_t dstSize) {
    uint64_t conversionCfg = 0;
    if (inputType != outputType) {
        if (inputType.isF32() && outputType.isF16()) {
            conversionCfg = DMA_DATA_CONV_FP32_FP16;
        } else if (inputType.isF32() && outputType.isBF16()) {
            conversionCfg = DMA_DATA_CONV_FP32_BF16;
        } else {
            VPUX_THROW("Unsupported DMA data conversion");
        }

        VPUX_THROW_WHEN(dstSize != (srcSize / 2), "Source and destination length do not match");
    }
    VPURegMapped::updateRegMappedInitializationValues(
            initValues,
            {
                    {"dma_cfg_fields",
                     {
                             {"dma_cfg_fields_conversion_cfg",
                              checked_cast_reg<NPUReg40XX::RegField_dma_cfg_fields_conversion_cfgType>(conversionCfg)},
                     }},
            });
}

uint32_t getActCompressionEntryTileMask(VPUASM::NNDMAOp dmaOp) {
    auto actCompressionSizeEntry = dmaOp.getActCompressionSizeEntry();
    if (actCompressionSizeEntry.has_value()) {
        auto actCompBufferRef = mlir::SymbolTable::lookupNearestSymbolFrom(dmaOp, actCompressionSizeEntry.value());
        VPUX_THROW_UNLESS(actCompBufferRef, "Could not find symbol name entry for {0} of {1}",
                          actCompressionSizeEntry.value(), dmaOp);

        if (mlir::isa<VPUASM::DeclareBufferOp>(actCompBufferRef)) {
            auto actCompBuffer = mlir::cast<VPUASM::DeclareBufferOp>(actCompBufferRef);
            return NPUReg40XX::getTileSelectMaskForBuffer(actCompBuffer);
        }
    }
    return 0;
}

void setDMAAccelerationCompress(RegisterMap& initValues, VPUASM::NNDMAOp origOp, mlir::MemRefType inputType,
                                mlir::MemRefType outputType) {
    auto accCfg = checked_cast_reg<NPUReg40XX::RegField_dma_cfg_fields_acceleration_cfgType>(DMA_ACCEL_COMPRESS);
    auto dtype = checked_cast_reg<NPUReg40XX::RegField_dma_acc_info_compress_dtypeType>(
            getTensorMode<NPUReg40XX::RegField_dma_acc_info_compress_dtypeType>(inputType.getElementType()));
    auto remoteWidthSetEn = checked_cast_reg<NPUReg40XX::RegField_dma_cfg_fields_rws_enType>(
            origOp.getActCompressionSizeEntry().has_value());

    uint32_t dmaActCompEntryTileMask = getActCompressionEntryTileMask(origOp);

    const auto dmaDescriptor = origOp.getDmaDescriptor();
    const auto srcWidth = dmaDescriptor.getSrcWidth().getInt();
    const auto dstWidth = outputType.cast<vpux::NDTypeInterface>().getTotalAllocSize().count();

    VPURegMapped::updateRegMappedInitializationValues(
            initValues,
            {{"dma_cfg_fields",
              {
                      {"dma_cfg_fields_rws_en", remoteWidthSetEn},
                      {"dma_cfg_fields_acceleration_cfg", accCfg},
              }},
             {"dma_acc_info_compress",
              {
                      {"dma_acc_info_compress_dtype",
                       checked_cast_reg<NPUReg40XX::RegField_dma_acc_info_compress_dtypeType>(dtype)},
                      {"dma_acc_info_compress_bitc_en",
                       checked_cast_reg<NPUReg40XX::RegField_dma_acc_info_compress_bitc_enType>(uint64_t(1))},
              }},
             {"dma_width", {{"dma_width_src", srcWidth}, {"dma_width_dst", dstWidth}}},
             {"dma_remote_width_store", {{"dma_remote_width_store", dmaActCompEntryTileMask}}}});
}

void setDMAAccelerationDecompress(RegisterMap& initValues, VPUASM::NNDMAOp origOp, mlir::MemRefType outputType) {
    auto accCfg = checked_cast_reg<NPUReg40XX::RegField_dma_cfg_fields_acceleration_cfgType>(DMA_ACCEL_DECOMPRESS);
    auto dtype = checked_cast_reg<NPUReg40XX::RegField_dma_acc_info_decompress_dtypeType>(
            getTensorMode<NPUReg40XX::RegField_dma_acc_info_decompress_dtypeType>(outputType.getElementType()));
    auto actCompressionSizeEntry = origOp.getActCompressionSizeEntry();
    auto remoteWidthFetchEn =
            checked_cast_reg<NPUReg40XX::RegField_dma_cfg_fields_rwf_enType>(actCompressionSizeEntry.has_value());

    uint32_t dmaActDecompEntryTileMask = getActCompressionEntryTileMask(origOp);

    VPURegMapped::updateRegMappedInitializationValues(
            initValues,
            {{"dma_cfg_fields",
              {
                      {"dma_cfg_fields_rwf_en", remoteWidthFetchEn},
                      {"dma_cfg_fields_acceleration_cfg", accCfg},
              }},
             {"dma_acc_info_decompress",
              {
                      {"dma_acc_info_decompress_dtype",
                       checked_cast_reg<NPUReg40XX::RegField_dma_acc_info_decompress_dtypeType>(dtype)},
                      {"dma_acc_info_decompress_bitc_en",
                       checked_cast_reg<NPUReg40XX::RegField_dma_acc_info_decompress_bitc_enType>(uint64_t(1))},
              }},
             {"dma_remote_width_fetch", {{"dma_remote_width_fetch", dmaActDecompEntryTileMask}}}});
}

void setDMAAccelerationMode(RegisterMap& initValues, VPUASM::NNDMAOp origOp, mlir::MemRefType inputType,
                            mlir::MemRefType outputType) {
    auto accMode = origOp.getAccelerationMode();
    switch (accMode) {
    case VPUIP::DMAAccMode::DISABLE:
        // nothing to do
        break;
    case VPUIP::DMAAccMode::COMPRESSION:
        setDMAAccelerationCompress(initValues, origOp, inputType, outputType);
        break;
    case VPUIP::DMAAccMode::DECOMPRESSION:
        setDMAAccelerationDecompress(initValues, origOp, outputType);
        break;
    default:
        VPUX_THROW("{0} acceleration mode is not supported", accMode);
        break;
    }
}

void setEnableMemorySideCaching(RegisterMap& initValues) {
    auto srcMscMode = vpux::VPURegMapped::checked_cast_reg<vpux::NPUReg40XX::RegField_dma_src_aubType>(DMA_AUB_SRC_DST);
    auto dstMscMode = vpux::VPURegMapped::checked_cast_reg<vpux::NPUReg40XX::RegField_dma_dst_aubType>(DMA_AUB_SRC_DST);
    auto aubCfgMode =
            vpux::VPURegMapped::checked_cast_reg<vpux::NPUReg40XX::RegField_dma_cfg_fields_axi_user_bits_cfgType>(
                    DMA_AUB_SRC_DST);

    VPURegMapped::updateRegMappedInitializationValues(initValues,
                                                      {
                                                              {"dma_src_aub", {{"dma_src_aub", srcMscMode}}},
                                                              {"dma_dst_aub", {{"dma_dst_aub", dstMscMode}}},
                                                      });
    VPURegMapped::updateRegMappedInitializationValues(
            initValues, {{"dma_cfg_fields", {{"dma_cfg_fields_axi_user_bits_cfg", aubCfgMode}}}});
}

struct NPUReg40XX_3D_DmaConfig {
    uint64_t srcWidth;
    uint64_t dstWidth;

    uint64_t srcDimSize1;
    uint64_t dstDimSize1;

    uint64_t srcStride1;
    uint64_t dstStride1;

    uint64_t srcDimSize2;
    uint64_t dstDimSize2;

    uint64_t srcStride2;
    uint64_t dstStride2;

    uint64_t numDims;
};

NPUReg40XX_3D_DmaConfig configure3DDma(VPUIP::DMADescriptorAttr vpu27Config) {
    NPUReg40XX_3D_DmaConfig vpu4config{};

    auto numPlanes = vpu27Config.getNumPlanes().getInt();
    auto length = vpu27Config.getLen().getInt();
    auto srcWidth = vpu27Config.getSrcWidth().getInt();
    auto srcStride = vpu27Config.getSrcStride().getInt();
    auto srcPlaneStride = vpu27Config.getSrcPlaneStride().getInt();
    auto dstWidth = vpu27Config.getDstWidth().getInt();
    auto dstStride = vpu27Config.getDstStride().getInt();
    auto dstPlaneStride = vpu27Config.getDstPlaneStride().getInt();

    auto srcDimSize1 = srcWidth ? static_cast<int64_t>((length / srcWidth) - 1) : 0;
    auto dstDimSize1 = (dstWidth && (length > dstWidth)) ? static_cast<int64_t>((length / dstWidth) - 1) : 0;
    int64_t numDims = numPlanes <= 1 ? DMA_2D : DMA_3D;
    vpu4config.numDims = checked_cast_reg<NPUReg40XX::RegField_dma_cfg_fields_num_dimType>(numDims);
    vpu4config.srcWidth = checked_cast_reg<NPUReg40XX::RegField_dma_width_srcType>(srcWidth);
    vpu4config.dstWidth = checked_cast_reg<NPUReg40XX::RegField_dma_width_dstType>(dstWidth);

    switch (numDims) {
    case DMA_3D:
        VPUX_THROW_WHEN(numPlanes == 0, "numPlanes cannot be 0 for a 3D transaction");
        vpu4config.srcDimSize2 = checked_cast_reg<NPUReg40XX::RegField_dma_dim_size_2_srcType>(numPlanes - 1);
        vpu4config.dstDimSize2 = checked_cast_reg<NPUReg40XX::RegField_dma_dim_size_2_dstType>(numPlanes - 1);

        vpu4config.srcStride2 = checked_cast_reg<NPUReg40XX::RegField_dma_stride_src_2Type>(srcPlaneStride);
        vpu4config.dstStride2 = checked_cast_reg<NPUReg40XX::RegField_dma_stride_dst_2Type>(dstPlaneStride);

        [[fallthrough]];
    case DMA_2D:
        vpu4config.srcDimSize1 = checked_cast_reg<NPUReg40XX::RegField_dma_dim_size_1_srcType>(srcDimSize1);
        vpu4config.dstDimSize1 = checked_cast_reg<NPUReg40XX::RegField_dma_dim_size_1_dstType>(dstDimSize1);

        vpu4config.srcStride1 = checked_cast_reg<NPUReg40XX::RegField_dma_stride_src_1Type>(srcStride);
        vpu4config.dstStride1 = checked_cast_reg<NPUReg40XX::RegField_dma_stride_dst_1Type>(dstStride);
        break;
    default:
        VPUX_THROW("Error at configure3DDma. Unsupported numDims={0}", numDims);
        break;
    }

    return vpu4config;
}

class NNDMARewriter final : public mlir::OpRewritePattern<VPUASM::NNDMAOp> {
public:
    NNDMARewriter(mlir::MLIRContext* ctx, Logger log, ELF::SymbolReferenceMap& symRefMap)
            : mlir::OpRewritePattern<VPUASM::NNDMAOp>(ctx), _log(log), _symRefMap(symRefMap) {
        setDebugName("NNDMA_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::NNDMAOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isWorkLoadManagementDMA(mlir::Operation* op) const {
        if (mlir::isa<VPUASM::DPUInvariantOp>(op) || mlir::isa<VPUASM::DPUVariantOp>(op) ||
            mlir::isa<VPUIPDPU::DPUInvariantOp>(op) || mlir::isa<VPUIPDPU::DPUVariantOp>(op) ||
            mlir::isa<VPUASM::ActKernelInvocationOp>(op) || mlir::isa<VPUASM::ActKernelRangeOp>(op) ||
            mlir::isa<VPUASM::DeclareTaskBufferOp>(op)) {
            return true;
        }
        return false;
    }
    Logger _log;
    ELF::SymbolReferenceMap& _symRefMap;
};

// Hardware supports Gather/Scatter mode, currently only Gather is supported by compiler.
void setGatherMode(ELF::SymbolReferenceMap& _symRefMap, const ::mlir::SymbolRefAttr& indices, RegisterMap& initValues,
                   const mlir::MemRefType& outputType, Bit elemOutSize) {
    mlir::MemRefType indicesType;
    auto indicesBufferRep = _symRefMap.lookupSymbol(indices);
    if (mlir::isa<VPUASM::DeclareBufferOp>(indicesBufferRep)) {
        auto indicesBuffer = mlir::cast<VPUASM::DeclareBufferOp>(indicesBufferRep);
        indicesType = indicesBuffer.getBufferType().getMemref();
    }
    // DMA copies data block by block example here with
    // input 50257x768xf32 indices 1024xi64
    // output 1024x768xf32 so dma will copy 1024 blocks of 768xsizeof(f32) blocks so elementsize here would be 3072
    // We dont need axis info here as gather cant be done here if indices are pointing so leftmost dimension who has
    // size bigger than 1. There condition will be enforced on upper level dialects.
    const auto dma_element_size =
            (outputType.getNumElements() / indicesType.getNumElements()) * elemOutSize.to<Byte>().count();
    VPURegMapped::updateRegMappedInitializationValues(
            initValues,
            {
                    {"dma_cfg_fields",
                     {{"dma_cfg_fields_src_list_cfg",
                       DMA_LIST_REL_INDEX},  // or DMA_LIST_ABS_INDEX - absolute or relative indexing
                      {"dma_cfg_fields_dst_list_cfg", 0}}},
                    {"dma_list_size",
                     {{"dma_list_size_src", indicesType.getNumElements()}}},  //  the number of indices in the list.
                    {"dma_stride_dst_1", {{"dma_stride_dst_1", dma_element_size}}},  // element_size_cmx_padded
                    {"dma_width", {{"dma_width_src", dma_element_size}}},            // element_size_ddr_packed
                    {"dma_dim_size", {{"dma_dim_size_1_dst", 0}}}
                    // Data written must be equal to data read, Src_width* Src_list_size = Dst_width*(Dst_dim_size[1]+1)
                    // Dst_width = Src_width* Src_list_size so Dst_dim_size[1] should be always 0 (This is GatherDMA
                    // specific)
            });
}

mlir::LogicalResult NNDMARewriter::matchAndRewrite(VPUASM::NNDMAOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // VPUASM ops already contain information about input/output buffers in `dma_descriptor` field
    // we should use it instead of looking related memref's by sym names
    // TODO: E#73178
    auto inputBufferRef = _symRefMap.lookupSymbol(origOp.getInput());
    VPUX_THROW_UNLESS(inputBufferRef, "Could not find symbol name entry for {0} of {1}", origOp.getInput(), origOp);
    mlir::MemRefType inputType;

    uint32_t inputTileMask = 0;
    bool isWLMNNDMAOp = false;
    if (mlir::isa<VPUASM::DeclareBufferOp>(inputBufferRef)) {
        auto inputBuffer = mlir::cast<VPUASM::DeclareBufferOp>(inputBufferRef);
        inputType = inputBuffer.getBufferType().getMemref();
        inputTileMask = vpux::NPUReg40XX::getTileSelectMaskForBuffer(inputBuffer);
    } else if (mlir::isa<VPUASM::ConstBufferOp>(inputBufferRef)) {
        auto inputBuffer = mlir::cast<VPUASM::ConstBufferOp>(inputBufferRef);
        inputType = inputBuffer.getBufferType().getMemref();
    } else if (isWorkLoadManagementDMA(inputBufferRef)) {
        isWLMNNDMAOp = true;
    } else {
        VPUX_THROW("Could not find symbol name entry for {0}", origOp.getInput());
    }

    uint32_t broadcastTileMask = 0;
    if (origOp.getTileIndexes().has_value()) {
        for (auto buffIndex : parseIntArrayAttr<int64_t>(origOp.getTileIndexes().value())) {
            broadcastTileMask |=
                    1 << (buffIndex + NPUReg40XX::CMX_TILE_SELECT_OFFSET);  // Bits 21 to 26 are used for tile select
        }
    }

    auto prodMaskLo = checked_cast_reg<NPUReg40XX::RegField_dma_barrier_prod_mask_lowerType>(
            vpux::VPUMI40XX::computeMaskLo(origOp.getUpdateBarriers()));
    auto prodMaskHi = checked_cast_reg<NPUReg40XX::RegField_dma_barrier_prod_mask_upperType>(
            vpux::VPUMI40XX::computeMaskHi(origOp.getUpdateBarriers()));
    auto consMaskLo = checked_cast_reg<NPUReg40XX::RegField_dma_barrier_cons_mask_lowerType>(
            vpux::VPUMI40XX::computeMaskLo(origOp.getWaitBarriers()));
    auto consMaskHi = checked_cast_reg<NPUReg40XX::RegField_dma_barrier_cons_mask_upperType>(
            vpux::VPUMI40XX::computeMaskHi(origOp.getWaitBarriers()));

    const int barrierEn = 1;
    const int ord = !origOp.getIsOutOfOrder();

    uint32_t linkAddressTileMask = 0;
    if (origOp.getNextLink().has_value()) {
        auto nextDMARef = _symRefMap.lookupSymbol(origOp.getNextLink().value());

        auto nextDMATaskBuffer = mlir::dyn_cast<VPUASM::DeclareTaskBufferOp>(nextDMARef);
        if (nextDMATaskBuffer) {
            linkAddressTileMask = NPUReg40XX::getTileSelectMaskForBuffer(nextDMATaskBuffer);
        }
    }

    auto vpu4config = configure3DDma(origOp.getDmaDescriptor());

    auto startAfter = checked_cast_reg<NPUReg40XX::RegField_start_after_Type>(origOp.getStartAfter());
    auto cleanAfter = checked_cast_reg<NPUReg40XX::RegField_clean_after_Type>(origOp.getCleanAfter());

    // prepare DMARegister
    auto initValues = vpux::NPUReg40XX::RegMapped_DMARegisterType::getResetInitilizationValues();

    VPURegMapped::updateRegMappedInitializationValues(
            initValues,
            {
                    {"dma_cfg_fields",
                     {{"dma_cfg_fields_num_dim", vpu4config.numDims},
                      {"dma_cfg_fields_barrier_en", barrierEn},
                      {"dma_cfg_fields_atp_en", 1},
                      {"dma_cfg_fields_src_burst_length", 15},
                      {"dma_cfg_fields_dst_burst_length", 15},
                      {"dma_cfg_fields_arb_qos", 255},
                      {"dma_cfg_fields_ord", ord},
                      {"dma_cfg_fields_hwp_id_en", 1},
                      {"dma_cfg_fields_hwp_id", origOp.getDmaHwpId().value_or(0)}}},
                    {"dma_dim_size",
                     {{"dma_dim_size_1_src", vpu4config.srcDimSize1}, {"dma_dim_size_1_dst", vpu4config.dstDimSize1}}},
                    {"dma_stride_src_1", {{"dma_stride_src_1", vpu4config.srcStride1}}},
                    {"dma_stride_dst_1", {{"dma_stride_dst_1", vpu4config.dstStride1}}},
                    {"dma_dim_size_2",
                     {{"dma_dim_size_2_src", vpu4config.srcDimSize2}, {"dma_dim_size_2_dst", vpu4config.dstDimSize2}}},
                    {"dma_stride_src_2", {{"dma_stride_src_2", vpu4config.srcStride2}}},
                    {"dma_src_addr", {{"dma_src", inputTileMask}}},
                    {"dma_dst_addr", {{"dma_dst", broadcastTileMask}}},
                    {"dma_barrier_prod_mask_lower", {{"dma_barrier_prod_mask_lower", prodMaskLo}}},
                    {"dma_barrier_cons_mask_lower", {{"dma_barrier_cons_mask_lower", consMaskLo}}},
                    {"dma_barrier_prod_mask_upper", {{"dma_barrier_prod_mask_upper", prodMaskHi}}},
                    {"dma_barrier_cons_mask_upper", {{"dma_barrier_cons_mask_upper", consMaskHi}}},
                    {"dma_link_address", {{"dma_link_address", linkAddressTileMask}}},
                    {"dma_barriers_sched", {{"start_after_", startAfter}, {"clean_after_", cleanAfter}}},
            });

    if (auto enableMemorySideCaching = origOp.getEnableMscAttr()) {
        setEnableMemorySideCaching(initValues);
    }
    auto accMode = origOp.getAccelerationMode();
    auto actCompFlag = origOp.getActCompressionSizeEntry().has_value();
    auto indices = origOp.getIndices();
    // Some registers are conflicting with compression related registers and in such case they should not be programmed
    if (!actCompFlag) {
        // dma_width register conflicts with remote_width_fetch and should not be programmed in case of decompression
        // In case of compression it should not be programmed because dstWidth requires adjustment for worst case size
        VPURegMapped::updateRegMappedInitializationValues(
                initValues, {{"dma_width",
                              {{"dma_width_src", vpu4config.srcWidth},
                               {"dma_width_dst", vpu4config.dstWidth}}}});  // conflict with remote_width_fetch
    }

    if (!actCompFlag || accMode != vpux::VPUIP::DMAAccMode::COMPRESSION) {
        // dma_stride_dst_2 register conflicts with remote_width_store and should not be programmed in case of
        // compression
        VPURegMapped::updateRegMappedInitializationValues(
                initValues, {{"dma_stride_dst_2",
                              {{"dma_stride_dst_2", vpu4config.dstStride2}}}});  // conflict with remote_width_store
    }

    if (!isWLMNNDMAOp) {
        auto outputBufferSym = origOp.getOutputBuffs()[0].dyn_cast_or_null<mlir::SymbolRefAttr>();
        VPUX_THROW_UNLESS(outputBufferSym, "`output_buffs` attribute should contain SymbolRefAttr but it doesn't");

        auto outputBufferRef = _symRefMap.lookupSymbol(outputBufferSym);
        auto outputBuffer = mlir::dyn_cast_or_null<VPUASM::DeclareBufferOp>(outputBufferRef);
        VPUX_THROW_UNLESS(outputBuffer, "Could not find symbol name entry for {0}", outputBufferRef);
        auto outputType = outputBuffer.getBufferType().getMemref();

        const Bit elemInSize = vpux::getElemTypeSize(inputType);
        const Bit elemOutSize = vpux::getElemTypeSize(outputType);

        auto totalInSizeBits = alignMemSize(inputType.getNumElements() * elemInSize, Byte(1));
        auto totalOutSizeBits = alignMemSize(outputType.getNumElements() * elemOutSize, Byte(1));
        auto srcWidthSize =
                checked_cast_reg<NPUReg40XX::RegField_dma_width_srcType>(totalInSizeBits.to<Byte>().count());
        auto dstWidthSize =
                checked_cast_reg<NPUReg40XX::RegField_dma_width_dstType>(totalOutSizeBits.to<Byte>().count());

        // DMA only does FP32 -> FP16/BF16 conversions,
        // Because of this, dstDimSize1 will always be half of the original value
        if (inputType.getElementType() != outputType.getElementType() && vpu4config.dstDimSize1) {
            long newDstDimSize1 = ((vpu4config.dstDimSize1 + 1) / 2) - 1;
            VPURegMapped::updateRegMappedInitializationValues(
                    initValues,
                    {{"dma_dim_size", {{"dma_dim_size_1_dst", newDstDimSize1}}}});  // update value for conversion mode
        }

        if (accMode != vpux::VPUIP::DMAAccMode::DISABLE) {
            VPUX_THROW_WHEN(vpu4config.srcDimSize1 != 0 || vpu4config.dstDimSize1 != 0 || vpu4config.srcDimSize2 != 0 ||
                                    vpu4config.dstDimSize2 != 0 || vpu4config.srcStride2 != 0 ||
                                    vpu4config.dstStride2 != 0,
                            "Activation compression is supported only for 1D DMAs");
            setDMAAccelerationMode(initValues, origOp, inputType, outputType);
        } else {
            // Convertion
            setDMAConversionMode(initValues, inputType.getElementType(), srcWidthSize, outputType.getElementType(),
                                 dstWidthSize);
        }
        if (indices.has_value()) {
            setGatherMode(_symRefMap, indices.value(), initValues, outputType, elemOutSize);
        }
    }

    auto regDMADescriptorAttr =
            VPURegMapped::getRegMappedAttributeWithValues<NPUReg40XX::RegMapped_DMARegisterType>(rewriter, initValues);

    auto dma = rewriter.create<NPUReg40XX::NNDMAOp>(origOp->getLoc(), origOp.getSymNameAttr(), regDMADescriptorAttr,
                                                    origOp.getInputAttr(), origOp.getOutputBuffsAttr(),
                                                    origOp.getNextLinkAttr(), origOp.getActCompressionSizeEntryAttr(),
                                                    origOp.getIndicesAttr());

    // TODO: (E#114625) Remove once proper refactoring happened
    if (!origOp.getTaskLocationAttr()) {
        dma.getOperation()->setAttr("directLink", rewriter.getUnitAttr());
    }

    rewriter.eraseOp(origOp);

    return mlir::success();
}

void setNormFactor(RegisterMap& initValues, ::mlir::ArrayAttr normFactor) {
    const auto getRawFP16 = [](auto val) {
        const auto valFP16 = vpux::type::float16(val);
        return valFP16.to_bits();
    };

    auto normArr = parseFPArrayAttr<double>(normFactor);
    VPUX_THROW_UNLESS(normArr.size() == MEDIA_MAX_NUM_PLANES * 4 /*MEDIA_MAX_NUM_NORM_FACT*/,
                      "Normalization array is invalid");

    auto normFact00 = checked_cast_reg<NPUReg40XX::RegField_NormFact0Type>(getRawFP16(normArr[0]));
    auto normFact01 = checked_cast_reg<NPUReg40XX::RegField_NormFact1Type>(getRawFP16(normArr[1]));
    auto normFact02 = checked_cast_reg<NPUReg40XX::RegField_NormFact2Type>(getRawFP16(normArr[2]));
    auto normFact03 = checked_cast_reg<NPUReg40XX::RegField_NormFact3Type>(getRawFP16(normArr[3]));

    auto normFact10 = checked_cast_reg<NPUReg40XX::RegField_NormFact0Type>(getRawFP16(normArr[4]));
    auto normFact11 = checked_cast_reg<NPUReg40XX::RegField_NormFact1Type>(getRawFP16(normArr[5]));
    auto normFact12 = checked_cast_reg<NPUReg40XX::RegField_NormFact2Type>(getRawFP16(normArr[6]));
    auto normFact13 = checked_cast_reg<NPUReg40XX::RegField_NormFact3Type>(getRawFP16(normArr[7]));

    auto normFact20 = checked_cast_reg<NPUReg40XX::RegField_NormFact0Type>(getRawFP16(normArr[8]));
    auto normFact21 = checked_cast_reg<NPUReg40XX::RegField_NormFact1Type>(getRawFP16(normArr[9]));
    auto normFact22 = checked_cast_reg<NPUReg40XX::RegField_NormFact2Type>(getRawFP16(normArr[10]));
    auto normFact23 = checked_cast_reg<NPUReg40XX::RegField_NormFact3Type>(getRawFP16(normArr[11]));

    VPURegMapped::updateRegMappedInitializationValues(initValues, {{"NormFactor_0",
                                                                    {{"NormFact0", normFact00},
                                                                     {"NormFact1", normFact01},
                                                                     {"NormFact2", normFact02},
                                                                     {"NormFact3", normFact03}}},
                                                                   {"NormFactor_1",
                                                                    {{"NormFact0", normFact10},
                                                                     {"NormFact1", normFact11},
                                                                     {"NormFact2", normFact12},
                                                                     {"NormFact3", normFact13}}},
                                                                   {"NormFactor_2",
                                                                    {{"NormFact0", normFact20},
                                                                     {"NormFact1", normFact21},
                                                                     {"NormFact2", normFact22},
                                                                     {"NormFact3", normFact23}}}});
}

uint8_t getBytesOfPackOfPixels(VPU::M2iColorFmt inFormat) {
    switch (inFormat) {
    case VPU::M2iColorFmt::PL_FP16_RGB:
    case VPU::M2iColorFmt::PL_FP16_YUV:
    case VPU::M2iColorFmt::SP_NV12_10:
    case VPU::M2iColorFmt::SP_P010:
        return 2;
    case VPU::M2iColorFmt::IL_RGB888:
        return 3;
    case VPU::M2iColorFmt::IL_RGB8888:
    case VPU::M2iColorFmt::IL_RGB30:
        return 4;
    default:
        return 1;
    };
}

void setMediaDimensions(VPUASM::DeclareBufferOp bufferOp, VPU::M2iColorFmt format, uint64_t& width, uint64_t& height) {
    auto elemShape = bufferOp.getBufferType().getMemref().cast<NDTypeInterface>().getShape();

    switch (format) {
    case VPU::M2iColorFmt::PL_YUV420_8:
    case VPU::M2iColorFmt::SP_NV12_8:  // dims[] = N(0),H(1),W(2),C(3)
        // H / 3 * 2 -- These YUV formats have a full sized Y plane, and weaved U,V values,
        // hence we need to extract the height of the Y plane from the concatenated height
        height = elemShape[Dims4D::Act::C] / 3 * 2;
        width = elemShape[Dims4D::Act::H];
        break;

    case VPU::M2iColorFmt::IL_RGB888:  // dims[] = N(0),H(1),W(2),C(3)
        height = elemShape[Dims4D::Act::C];
        width = elemShape[Dims4D::Act::H];
        break;

    case VPU::M2iColorFmt::PL_RGB24:     // dims[] = N(0),C(1),H(2),W(3)
    case VPU::M2iColorFmt::PL_FP16_RGB:  // dims[] = N(0),C(1),H(2),W(3)
        height = elemShape[Dims4D::Act::H];
        width = elemShape[Dims4D::Act::W];
        break;

    default:
        VPUX_THROW("{0} format is not supported", format);
        break;
    }
}

void setInSizeDescription(RegisterMap& initValues, VPU::M2iColorFmt inFormat, uint64_t width, uint64_t height,
                          uint64_t m2iIndex) {
    uint64_t inSize0_ls(0), PSOB_inPS(0), inSize1_width(0), inSize1_height(0);
    uint64_t inSize1_ls(0), inSize2_width(0), inSize2_height(0), inSize2_ls(0);

    auto inSize0_width = checked_cast_reg<NPUReg40XX::RegField_widthType>(width - 1);
    auto inSize0_height = checked_cast_reg<NPUReg40XX::RegField_heightType>(height - 1);

    auto inSize0_PID = checked_cast_reg<NPUReg40XX::RegField_pidType>(m2iIndex);

    switch (inFormat) {
    case VPU::M2iColorFmt::PL_RGB24:
    case VPU::M2iColorFmt::PL_YUV444_8:
        inSize0_ls = checked_cast_reg<NPUReg40XX::RegField_lsType>(width);
        PSOB_inPS = checked_cast_reg<NPUReg40XX::RegField_inPSType>(width * height);
        inSize1_width = checked_cast_reg<NPUReg40XX::RegField_widthType>(width - 1);
        inSize1_height = checked_cast_reg<NPUReg40XX::RegField_heightType>(height - 1);
        inSize1_ls = checked_cast_reg<NPUReg40XX::RegField_lsType>(width);
        inSize2_width = checked_cast_reg<NPUReg40XX::RegField_widthType>(width - 1);
        inSize2_height = checked_cast_reg<NPUReg40XX::RegField_heightType>(height - 1);
        inSize2_ls = checked_cast_reg<NPUReg40XX::RegField_lsType>(width);
        break;

    case VPU::M2iColorFmt::PL_FP16_RGB:
        inSize0_ls = checked_cast_reg<NPUReg40XX::RegField_lsType>(width * 2);
        PSOB_inPS = checked_cast_reg<NPUReg40XX::RegField_inPSType>(width * height * 2);
        inSize1_width = checked_cast_reg<NPUReg40XX::RegField_widthType>(width - 1);
        inSize1_height = checked_cast_reg<NPUReg40XX::RegField_heightType>(height - 1);
        inSize1_ls = checked_cast_reg<NPUReg40XX::RegField_lsType>(width * 2);
        inSize2_width = checked_cast_reg<NPUReg40XX::RegField_widthType>(width - 1);
        inSize2_height = checked_cast_reg<NPUReg40XX::RegField_heightType>(height - 1);
        inSize2_ls = checked_cast_reg<NPUReg40XX::RegField_lsType>(width * 2);
        break;

    case VPU::M2iColorFmt::PL_GRAY8:
        inSize0_ls = checked_cast_reg<NPUReg40XX::RegField_lsType>(width);
        PSOB_inPS = checked_cast_reg<NPUReg40XX::RegField_inPSType>(width * height);
        inSize1_width = checked_cast_reg<NPUReg40XX::RegField_widthType>(width - 1);
        inSize1_height = checked_cast_reg<NPUReg40XX::RegField_heightType>(height - 1);
        inSize1_ls = checked_cast_reg<NPUReg40XX::RegField_lsType>(width);
        inSize2_width = checked_cast_reg<NPUReg40XX::RegField_widthType>(width - 1);
        inSize2_height = checked_cast_reg<NPUReg40XX::RegField_heightType>(height - 1);
        inSize2_ls = checked_cast_reg<NPUReg40XX::RegField_lsType>(width);
        break;

    case VPU::M2iColorFmt::SP_NV12_8:
        inSize0_ls = checked_cast_reg<NPUReg40XX::RegField_lsType>(width);
        PSOB_inPS = checked_cast_reg<NPUReg40XX::RegField_inPSType>(width * height);
        inSize1_width = checked_cast_reg<NPUReg40XX::RegField_widthType>(width - 1);
        inSize1_height = checked_cast_reg<NPUReg40XX::RegField_heightType>(height / 2 - 1);
        inSize1_ls = checked_cast_reg<NPUReg40XX::RegField_lsType>(width);
        break;

    case VPU::M2iColorFmt::PL_YUV420_8:
        inSize0_ls = checked_cast_reg<NPUReg40XX::RegField_lsType>(width);
        PSOB_inPS = checked_cast_reg<NPUReg40XX::RegField_inPSType>(width * height);
        inSize1_width = checked_cast_reg<NPUReg40XX::RegField_widthType>(width / 2 - 1);
        inSize1_height = checked_cast_reg<NPUReg40XX::RegField_heightType>(height / 2 - 1);
        inSize1_ls = checked_cast_reg<NPUReg40XX::RegField_lsType>(width / 2);
        inSize2_width = checked_cast_reg<NPUReg40XX::RegField_widthType>(width / 2 - 1);
        inSize2_height = checked_cast_reg<NPUReg40XX::RegField_heightType>(height / 2 - 1);
        inSize2_ls = checked_cast_reg<NPUReg40XX::RegField_lsType>(width / 2);
        break;

    case VPU::M2iColorFmt::PL_YUV422_8:
        inSize0_ls = checked_cast_reg<NPUReg40XX::RegField_lsType>(width);
        PSOB_inPS = checked_cast_reg<NPUReg40XX::RegField_inPSType>(width * height);
        inSize1_width = checked_cast_reg<NPUReg40XX::RegField_widthType>(width / 2 - 1);
        inSize1_height = checked_cast_reg<NPUReg40XX::RegField_heightType>(height - 1);
        inSize1_ls = checked_cast_reg<NPUReg40XX::RegField_lsType>(width / 2);
        inSize2_width = checked_cast_reg<NPUReg40XX::RegField_widthType>(width / 2 - 1);
        inSize2_height = checked_cast_reg<NPUReg40XX::RegField_heightType>(height - 1);
        inSize2_ls = checked_cast_reg<NPUReg40XX::RegField_lsType>(width / 2);
        break;

    case VPU::M2iColorFmt::IL_RGB888:
        inSize0_ls = checked_cast_reg<NPUReg40XX::RegField_lsType>(width * 3);
        PSOB_inPS = checked_cast_reg<NPUReg40XX::RegField_inPSType>(width * height * 3);
        inSize1_width = checked_cast_reg<NPUReg40XX::RegField_widthType>(width - 1);
        inSize1_height = checked_cast_reg<NPUReg40XX::RegField_heightType>(height - 1);
        inSize1_ls = checked_cast_reg<NPUReg40XX::RegField_lsType>(width * 3);
        inSize2_width = checked_cast_reg<NPUReg40XX::RegField_widthType>(width - 1);
        inSize2_height = checked_cast_reg<NPUReg40XX::RegField_heightType>(height - 1);
        inSize2_ls = checked_cast_reg<NPUReg40XX::RegField_lsType>(width * 3);
        break;

    default:
        VPUX_THROW("invalid input format {0}", inFormat);
        break;
    }

    VPURegMapped::updateRegMappedInitializationValues(
            initValues,
            {{"inSize0",
              {{"ls", inSize0_ls}, {"width", inSize0_width}, {"height", inSize0_height}, {"pid", inSize0_PID}}},
             {"inSize1", {{"ls", inSize1_ls}, {"width", inSize1_width}, {"height", inSize1_height}}},
             {"inSize2", {{"ls", inSize2_ls}, {"width", inSize2_width}, {"height", inSize2_height}}},
             {"PSOB", {{"inPS", PSOB_inPS}}}});
}

void setOutDescription(RegisterMap& initValues, VPU::M2iColorFmt outFormat, uint64_t outWidth, uint64_t outHeight) {
    uint64_t outScale0_width(0), outScale0_height(0);
    uint64_t psSc0Y(0), psSc0UV(0), lsSc0Y(0), lsSc0UV(0);

    switch (outFormat) {
    case VPU::M2iColorFmt::PL_RGB24:
    case VPU::M2iColorFmt::PL_GRAY8:
        outScale0_width = checked_cast_reg<NPUReg40XX::RegField_outScale0_widthType>(outWidth - 1);
        outScale0_height = checked_cast_reg<NPUReg40XX::RegField_outScale0_heightType>(outHeight - 1);
        psSc0Y = checked_cast_reg<NPUReg40XX::RegField_psSc0YType>(outWidth * outHeight);
        lsSc0Y = checked_cast_reg<NPUReg40XX::RegField_lsSc0YType>(outWidth);
        break;

    case VPU::M2iColorFmt::PL_FP16_YUV:
    case VPU::M2iColorFmt::PL_FP16_RGB:
        outScale0_width = checked_cast_reg<NPUReg40XX::RegField_outScale0_widthType>(outWidth - 1);
        outScale0_height = checked_cast_reg<NPUReg40XX::RegField_outScale0_heightType>(outHeight - 1);
        psSc0Y = checked_cast_reg<NPUReg40XX::RegField_psSc0YType>(outWidth * outHeight * 2);
        lsSc0Y = checked_cast_reg<NPUReg40XX::RegField_lsSc0YType>(outWidth * 2);
        break;

    case VPU::M2iColorFmt::SP_NV12_8:
        outScale0_width = checked_cast_reg<NPUReg40XX::RegField_outScale0_widthType>(outWidth - 1);
        outScale0_height = checked_cast_reg<NPUReg40XX::RegField_outScale0_heightType>(outHeight - 1);
        psSc0Y = checked_cast_reg<NPUReg40XX::RegField_psSc0YType>(outWidth * outHeight);
        psSc0UV = checked_cast_reg<NPUReg40XX::RegField_psSc0UVType>(outWidth * outHeight / 2);
        lsSc0Y = checked_cast_reg<NPUReg40XX::RegField_lsSc0YType>(outWidth);
        lsSc0UV = checked_cast_reg<NPUReg40XX::RegField_lsSc0UVType>(outWidth);
        break;

    case VPU::M2iColorFmt::PL_YUV420_8:
        outScale0_width = checked_cast_reg<NPUReg40XX::RegField_outScale0_widthType>(outWidth - 1);
        outScale0_height = checked_cast_reg<NPUReg40XX::RegField_outScale0_heightType>(outHeight - 1);
        psSc0Y = checked_cast_reg<NPUReg40XX::RegField_psSc0YType>(outWidth * outHeight);
        psSc0UV = checked_cast_reg<NPUReg40XX::RegField_psSc0UVType>(outWidth * outHeight / 4);
        lsSc0Y = checked_cast_reg<NPUReg40XX::RegField_lsSc0YType>(outWidth);
        lsSc0UV = checked_cast_reg<NPUReg40XX::RegField_lsSc0UVType>(outWidth / 2);
        break;

    case VPU::M2iColorFmt::PL_YUV422_8:
        outScale0_width = checked_cast_reg<NPUReg40XX::RegField_outScale0_widthType>(outWidth - 1);
        outScale0_height = checked_cast_reg<NPUReg40XX::RegField_outScale0_heightType>(outHeight - 1);
        psSc0Y = checked_cast_reg<NPUReg40XX::RegField_psSc0YType>(outWidth * outHeight);
        psSc0UV = checked_cast_reg<NPUReg40XX::RegField_psSc0UVType>(outWidth * outHeight / 2);
        lsSc0Y = checked_cast_reg<NPUReg40XX::RegField_lsSc0YType>(outWidth);
        lsSc0UV = checked_cast_reg<NPUReg40XX::RegField_lsSc0UVType>(outWidth / 2);
        break;

    case VPU::M2iColorFmt::PL_YUV444_8:
        outScale0_width = checked_cast_reg<NPUReg40XX::RegField_outScale0_widthType>(outWidth - 1);
        outScale0_height = checked_cast_reg<NPUReg40XX::RegField_outScale0_heightType>(outHeight - 1);
        psSc0Y = checked_cast_reg<NPUReg40XX::RegField_psSc0YType>(outWidth * outHeight);
        psSc0UV = checked_cast_reg<NPUReg40XX::RegField_psSc0UVType>(outWidth * outHeight);
        lsSc0Y = checked_cast_reg<NPUReg40XX::RegField_lsSc0YType>(outWidth);
        lsSc0UV = checked_cast_reg<NPUReg40XX::RegField_lsSc0UVType>(outWidth);
        break;

    case VPU::M2iColorFmt::IL_RGB888:
        outScale0_width = checked_cast_reg<NPUReg40XX::RegField_outScale0_widthType>(outWidth - 1);
        outScale0_height = checked_cast_reg<NPUReg40XX::RegField_outScale0_heightType>(outHeight - 1);
        psSc0Y = checked_cast_reg<NPUReg40XX::RegField_psSc0YType>(outWidth * outHeight * 3);
        lsSc0Y = checked_cast_reg<NPUReg40XX::RegField_lsSc0YType>(outWidth * 3);
        break;

    default:
        VPUX_THROW("invalid output format {0}", outFormat);
        break;
    }

    VPURegMapped::updateRegMappedInitializationValues(
            initValues,
            {{"OutScaleSize", {{"outScale0_width", outScale0_width}, {"outScale0_height", outScale0_height}}},
             {"ScPSY", {{"psSc0Y", psSc0Y}}},
             {"ScPSUV", {{"psSc0UV", psSc0UV}}},
             {"OutLS", {{"lsSc0Y", lsSc0Y}, {"lsSc0UV", lsSc0UV}}}});
}

bool isCscRequired(VPU::M2iColorFmt inFormat, VPU::M2iColorFmt outFormat) {
    // Automatically switch CSC on when input format and output format are different
    // and they are found in a viable conversion list
    llvm::DenseMap<VPU::M2iColorFmt, llvm::DenseSet<VPU::M2iColorFmt>> supportedInOutFormatMap = {
            {VPU::M2iColorFmt::SP_NV12_8,
             {VPU::M2iColorFmt::PL_RGB24, VPU::M2iColorFmt::IL_RGB888, VPU::M2iColorFmt::PL_FP16_RGB}},
            {VPU::M2iColorFmt::PL_RGB24,
             {VPU::M2iColorFmt::SP_NV12_8, VPU::M2iColorFmt::PL_YUV444_8, VPU::M2iColorFmt::PL_YUV422_8,
              VPU::M2iColorFmt::PL_GRAY8, VPU::M2iColorFmt::PL_YUV420_8}},
            {VPU::M2iColorFmt::IL_RGB888, {VPU::M2iColorFmt::SP_NV12_8}},
            {VPU::M2iColorFmt::PL_YUV444_8, {VPU::M2iColorFmt::PL_RGB24}},
            {VPU::M2iColorFmt::PL_YUV422_8, {VPU::M2iColorFmt::PL_RGB24}},
            {VPU::M2iColorFmt::PL_YUV420_8,
             {VPU::M2iColorFmt::PL_RGB24, VPU::M2iColorFmt::IL_RGB888, VPU::M2iColorFmt::PL_FP16_RGB}}};

    return (supportedInOutFormatMap.find(inFormat) != supportedInOutFormatMap.end() &&
            supportedInOutFormatMap[inFormat].find(outFormat) != supportedInOutFormatMap[inFormat].end());
}

class M2IRewriter final : public mlir::OpRewritePattern<VPUASM::M2IOp> {
public:
    M2IRewriter(mlir::MLIRContext* ctx, Logger log, ELF::SymbolReferenceMap& symRefMap)
            : mlir::OpRewritePattern<VPUASM::M2IOp>(ctx), _log(log), _symRefMap(symRefMap) {
        setDebugName("M2I_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::M2IOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    ELF::SymbolReferenceMap& _symRefMap;
};

mlir::LogicalResult M2IRewriter::matchAndRewrite(VPUASM::M2IOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    // prepare MediaRegister
    auto initValues = NPUReg40XX::RegMapped_VpuMediaTaskType::getResetInitilizationValues();

    auto inFormat = checked_cast_reg<NPUReg40XX::RegField_inFormatType>(origOp.getInFmt());
    auto outFormat = checked_cast_reg<NPUReg40XX::RegField_outFormatType>(origOp.getOutFmt());
    auto sampleType = checked_cast_reg<NPUReg40XX::RegField_sampleTypeType>(origOp.getInterp());

    const auto chromaInRC = static_cast<uint64_t>(origOp.getChromaInReverseChannels());
    const auto lumaInRC = static_cast<uint64_t>(origOp.getLumaInReverseChannels());
    auto ifc = checked_cast_reg<NPUReg40XX::RegField_IFCType>(((chromaInRC & 0x1) << 5) | ((lumaInRC & 0x1) << 4) |
                                                              (getBytesOfPackOfPixels(origOp.getInFmt()) & 0xF));
    uint64_t irqMask = 1 << 15;
    irqMask = checked_cast_reg<NPUReg40XX::RegField_IRQMaskType>(irqMask);

    auto hScOffset = checked_cast_reg<NPUReg40XX::RegField_hSc_offsetType>(origOp.getTileOffsetX().value_or(0));
    auto hScFactor = checked_cast_reg<NPUReg40XX::RegField_hSc_factorType>(origOp.getScaleFactorX());
    auto vScOffset = checked_cast_reg<NPUReg40XX::RegField_hSc_offsetType>(origOp.getTileOffsetY().value_or(0));
    auto vScFactor = checked_cast_reg<NPUReg40XX::RegField_hSc_factorType>(origOp.getScaleFactorY());

    const auto chromaOutRC = static_cast<uint64_t>(origOp.getChromaOutReverseChannels());
    const auto lumaOutRC = static_cast<uint64_t>(origOp.getLumaOutReverseChannels());
    auto ofc = checked_cast_reg<NPUReg40XX::RegField_OFCType>(((chromaOutRC & 0x1) << 1) | (lumaOutRC & 0x1));

    uint64_t nextDescTileMask = 0;
    if (origOp.getNextLink().has_value()) {
        auto nextM2IRef = _symRefMap.lookupSymbol(origOp.getNextLink().value());
        if (auto nextM2ITaskBuffer = mlir::dyn_cast<VPUASM::DeclareTaskBufferOp>(nextM2IRef)) {
            nextDescTileMask = NPUReg40XX::getTileSelectMaskForBuffer(nextM2ITaskBuffer);
        }
    }

    uint64_t width(0), height(0), inputTileMask(0);
    auto m2iIndex = origOp.getTaskIndex().getValue();
    auto inBufferRef = _symRefMap.lookupSymbol(origOp.getInput());
    auto inBufferOp = mlir::dyn_cast_or_null<VPUASM::DeclareBufferOp>(inBufferRef);
    VPUX_THROW_UNLESS(inBufferOp, "Could not find symbol name entry for {0}", inBufferRef);
    inputTileMask = NPUReg40XX::getTileSelectMaskForBuffer(inBufferOp);
    setMediaDimensions(inBufferOp, origOp.getInFmt(), width, height);
    setInSizeDescription(initValues, origOp.getInFmt(), width, height, m2iIndex);
    auto roiWidth = checked_cast_reg<NPUReg40XX::RegField_roiWidthType>(width - 1);
    auto roiHeight = checked_cast_reg<NPUReg40XX::RegField_roiHeightType>(height - 1);

    uint64_t outWidth(0), outHeight(0), outputTileMask(0);
    auto outBufferRef = _symRefMap.lookupSymbol(origOp.getOutputBuff());
    auto outBufferOp = mlir::dyn_cast_or_null<VPUASM::DeclareBufferOp>(outBufferRef);
    VPUX_THROW_UNLESS(outBufferOp, "Could not find symbol name entry for {0}", outBufferRef);
    outputTileMask = NPUReg40XX::getTileSelectMaskForBuffer(outBufferOp);
    setMediaDimensions(outBufferOp, origOp.getOutFmt(), outWidth, outHeight);
    setOutDescription(initValues, origOp.getOutFmt(), outWidth, outHeight);
    outWidth = checked_cast_reg<NPUReg40XX::RegField_outScale0_widthType>(outWidth - 1);
    outHeight = checked_cast_reg<NPUReg40XX::RegField_outScale0_heightType>(outHeight - 1);

    if (origOp.getNorm().has_value()) {
        setNormFactor(initValues, origOp.getNorm().value());
    }

    auto startAfter = checked_cast_reg<NPUReg40XX::RegField_start_after_Type>(origOp.getStartAfter());
    auto cleanAfter = checked_cast_reg<NPUReg40XX::RegField_clean_after_Type>(origOp.getCleanAfter());

    auto barGateMaskLO = checked_cast_reg<NPUReg40XX::RegField_barGateMaskLOType>(
            VPUMI40XX::computeMaskLo(origOp.getWaitBarriers()));
    auto barGateMaskHI = checked_cast_reg<NPUReg40XX::RegField_barGateMaskHIType>(
            VPUMI40XX::computeMaskHi(origOp.getWaitBarriers()));
    auto updateLO = checked_cast_reg<NPUReg40XX::RegField_barUpdateLOType>(
            VPUMI40XX::computeMaskLo(origOp.getUpdateBarriers()));
    auto updatekHi = checked_cast_reg<NPUReg40XX::RegField_barUpdateHIType>(
            VPUMI40XX::computeMaskHi(origOp.getUpdateBarriers()));

    outFormat = checked_cast_reg<NPUReg40XX::RegField_outFormatLocalType>(outFormat);
    sampleType = checked_cast_reg<NPUReg40XX::RegField_samlingTypeLocalType>(sampleType);

    uint64_t operations(0);
    operations |= origOp.getDoCsc() ? (1 << 0) : 0;
    operations |= isCscRequired(origOp.getInFmt(), origOp.getOutFmt()) ? (1 << 3 | 1 << 0) : 0;  // CLAMP bit always set
    operations |= origOp.getDoNorm() ? 1 << 1 : 0;
    operations = checked_cast_reg<NPUReg40XX::RegField_operationsType>(operations);

    VPURegMapped::updateRegMappedInitializationValues(
            initValues, {
                                {"inAddr0", {{"inAddr", inputTileMask}}},
                                {"inAddr1", {{"inAddr", inputTileMask}}},
                                {"inAddr2", {{"inAddr", inputTileMask}}},
                                {"IOCfg",
                                 {{"inFormat", inFormat},
                                  {"outFormat", outFormat},
                                  {"sampleType", sampleType},
                                  {"numRois", 1},
                                  {"IFC", ifc},
                                  {"IRQMask", irqMask},
                                  {"operations", operations}}},
                                {"RoiDef",
                                 {{"roiBase", outputTileMask},
                                  {"OFC", ofc},
                                  {"outFormatLocal", outFormat},
                                  {"samlingTypeLocal", sampleType}}},
                                {"OutScaleSize", {{"outScale0_width", outWidth}, {"outScale0_height", outHeight}}},
                                {"RoiCfg", {{"roiWidth", roiWidth}, {"roiHeight", roiHeight}}},
                                {"ScOffset", {{"vSc_offset", vScOffset}, {"hSc_offset", hScOffset}}},
                                {"ScFactor", {{"vSc_factor", vScFactor}, {"hSc_factor", hScFactor}}},
                                {"nextDesc", {{"nextDesc", nextDescTileMask}}},
                                {"barGateMaskLO", {{"barGateMaskLO", barGateMaskLO}}},
                                {"barGateMaskHI", {{"barGateMaskHI", barGateMaskHI}}},
                                {"barUpdateLO", {{"barUpdateLO", updateLO}}},
                                {"barUpdateHI", {{"barUpdateHI", updatekHi}}},
                                {"media_barriers_sched_", {{"start_after_", startAfter}, {"clean_after_", cleanAfter}}},
                        });

    auto regM2IDescriptorAttr =
            VPURegMapped::getRegMappedAttributeWithValues<NPUReg40XX::RegMapped_VpuMediaTaskType>(rewriter, initValues);

    rewriter.create<NPUReg40XX::M2IOp>(origOp->getLoc(), origOp.getSymNameAttr(), origOp.getInputAttr(),
                                       origOp.getOutputBuffAttr(), origOp.getProfilingDataAttr(),
                                       origOp.getNextLinkAttr(), regM2IDescriptorAttr);

    rewriter.eraseOp(origOp);

    return mlir::success();
}

//
// ActShave
//

class ActShaveRtRewriter final : public mlir::OpRewritePattern<VPUASM::ActShaveRtOp> {
public:
    ActShaveRtRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUASM::ActShaveRtOp>(ctx), _log(log) {
        setDebugName("ActShaveRt_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::ActShaveRtOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ActShaveRtRewriter::matchAndRewrite(VPUASM::ActShaveRtOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    rewriter.create<NPUReg40XX::ActShaveRtOp>(origOp->getLoc(), origOp.getSymNameAttr(), origOp.getKernelPathAttr());
    rewriter.eraseOp(origOp);
    return mlir::success();
}

//
// ActKernelRange
//
class ActKernelRangeRewriter final : public mlir::OpRewritePattern<VPUASM::ActKernelRangeOp> {
public:
    ActKernelRangeRewriter(mlir::MLIRContext* ctx, Logger log, ELF::SymbolReferenceMap& symRefMap)
            : mlir::OpRewritePattern<VPUASM::ActKernelRangeOp>(ctx), _log(log), _symRefMap(symRefMap) {
        setDebugName("ActKernelRange_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::ActKernelRangeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    ELF::SymbolReferenceMap& _symRefMap;
};

mlir::LogicalResult ActKernelRangeRewriter::matchAndRewrite(VPUASM::ActKernelRangeOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto kernelEntry = NPUReg40XX::getKernelEntry(_symRefMap, origOp.getKernelEntry());
    auto kernelTextSize = NPUReg40XX::getKernelTextSize(_symRefMap, origOp.getKernelText());
    auto kernelTaskType = origOp.getKernelTaskType();
    auto kernelPath = NPUReg40XX::getKernelPath(_symRefMap, origOp.getKernelEntry(), kernelTaskType);
    auto actWLtype = static_cast<std::underlying_type<npu40xx::nn_public::VpuActWLType>::type>(
            NPUReg40XX::getActWLType(kernelTaskType));

    auto regActKernelDescriptorAttr =
            vpux::VPURegMapped::getRegMappedAttributeWithValues<vpux::NPUReg40XX::RegMapped_VpuActKernelRangeType>(
                    rewriter, {{"type", {{"type", actWLtype}}},
                               {"kernel_entry", {{"kernel_entry", kernelEntry}}},
                               {"code_size", {{"code_size", kernelTextSize}}}});

    rewriter.create<NPUReg40XX::ActKernelRangeOp>(origOp->getLoc(), origOp.getSymNameAttr(), regActKernelDescriptorAttr,
                                                  origOp.getTaskLocationAttr(), origOp.getKernelTextAttr(),
                                                  origOp.getKernelEntryAttr());

    _log.trace("[{0}] Got kernel '{1}' and cpu '{2}'", getDebugName(), kernelPath, VPU::getArch(origOp));

    rewriter.eraseOp(origOp);

    return mlir::success();
}

//
// ActKernelInvocation
//
class ActKernelInvocationRewriter final : public mlir::OpRewritePattern<VPUASM::ActKernelInvocationOp> {
public:
    ActKernelInvocationRewriter(mlir::MLIRContext* ctx, Logger log, ELF::SymbolReferenceMap& symRefMap)
            : mlir::OpRewritePattern<VPUASM::ActKernelInvocationOp>(ctx), _log(log), _symRefMap(symRefMap) {
        setDebugName("ActKernelInvocation_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::ActKernelInvocationOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    ELF::SymbolReferenceMap& _symRefMap;
};

mlir::LogicalResult ActKernelInvocationRewriter::matchAndRewrite(VPUASM::ActKernelInvocationOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto kernelRangeRef = _symRefMap.lookupSymbol(origOp.getKernelRange());
    auto kernelRangeTaskBufferOp = mlir::cast<VPUASM::DeclareTaskBufferOp>(kernelRangeRef);
    auto kernelRangeTileMask = NPUReg40XX::getTileSelectMaskForBuffer(kernelRangeTaskBufferOp);
    auto kernelRangeIndex = origOp.getRangeIndex();

    uint64_t perfPacketTileMask = 0;
    if (auto profilingDataOpt = origOp.getProfilingData()) {
        auto perfPacketBufferRef = _symRefMap.lookupSymbol(*profilingDataOpt);
        auto perfPacketBufferOp = mlir::cast<VPUASM::DeclareBufferOp>(perfPacketBufferRef);
        perfPacketTileMask = NPUReg40XX::getTileSelectMaskForBuffer(perfPacketBufferOp);
    }

    auto invoIndex = origOp.getTaskIndex().getValue();

    auto waitMaskHi = VPUMI40XX::computeMaskHi(origOp.getWaitBarriers());
    auto waitMaskLo = VPUMI40XX::computeMaskLo(origOp.getWaitBarriers());
    auto postMaskHi = VPUMI40XX::computeMaskHi(origOp.getUpdateBarriers());
    auto postMaskLo = VPUMI40XX::computeMaskLo(origOp.getUpdateBarriers());

    uint8_t barrier_group = 0;
    uint8_t barrier_mask = 0;

    std::tie(barrier_group, barrier_mask) = reduceWaitMaskTo8bit(waitMaskLo);

    auto regActKernelInvoDescriptorAttr =
            vpux::VPURegMapped::getRegMappedAttributeWithValues<vpux::NPUReg40XX::RegMapped_VpuActKernelInvocationType>(
                    rewriter, {{"range", {{"range", kernelRangeTileMask}}},
                               {"barriers_wait_mask_hi_act", {{"barriers_wait_mask_hi_act", waitMaskHi}}},
                               {"barriers_wait_mask_lo_act", {{"barriers_wait_mask_lo_act", waitMaskLo}}},
                               {"barriers_post_mask_hi_act", {{"barriers_post_mask_hi_act", postMaskHi}}},
                               {"barriers_post_mask_lo_act", {{"barriers_post_mask_lo_act", postMaskLo}}},
                               {"barriers_group_mask_act", {{"group_act", barrier_group}, {"mask_act", barrier_mask}}},
                               {"act_invo_barriers_sched",
                                {{"start_after_", origOp.getStartAfter()}, {"clean_after_", origOp.getCleanAfter()}}},
                               {"invo_index", {{"invo_index", invoIndex}}},
                               {"invo_tile", {{"invo_tile", origOp.getTile()}}},
                               {"kernel_range_index", {{"kernel_range_index", kernelRangeIndex}}},
                               {"perf_packet_out", {{"perf_packet_out", perfPacketTileMask}}}});

    rewriter.create<NPUReg40XX::ActKernelInvocationOp>(origOp->getLoc(), origOp.getSymNameAttr(),
                                                       regActKernelInvoDescriptorAttr, origOp.getTaskLocationAttr(),
                                                       origOp.getKernelRangeAttr(), origOp.getKernelDataAttr(),
                                                       origOp.getKernelParamsAttr(), origOp.getProfilingDataAttr());

    rewriter.eraseOp(origOp);

    return mlir::success();
}

//
// ConvertVPUASM2NPUReg40XXRelocsPass
//

class ConvertVPUASM2NPUReg40XXRelocsPass final :
        public ConvertVPUASM2NPUReg40XXRelocsBase<ConvertVPUASM2NPUReg40XXRelocsPass> {
public:
    explicit ConvertVPUASM2NPUReg40XXRelocsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    explicit ConvertVPUASM2NPUReg40XXRelocsPass(Logger log, bool enableWLM) {
        Base::initLogger(log, Base::getArgumentName());
        _enableWLM = enableWLM;
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnModule() final;
    bool _enableWLM;
};

mlir::LogicalResult ConvertVPUASM2NPUReg40XXRelocsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (wlmEnabled.hasValue()) {
        _enableWLM = wlmEnabled.getValue();
    }

    return mlir::success();
}

void ConvertVPUASM2NPUReg40XXRelocsPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto& ctx = getContext();
    mlir::func::FuncOp netFunc;
    IE::CNNNetworkOp cnnOp;

    IE::CNNNetworkOp::getFromModule(moduleOp, cnnOp, netFunc);

    mlir::ConversionTarget target(ctx);

    target.addLegalDialect<NPUReg40XX::NPUReg40XXDialect>();
    target.addLegalDialect<VPUASM::VPUASMDialect>();

    target.addIllegalOp<VPUASM::ActKernelInvocationOp>();
    target.addIllegalOp<VPUASM::ActKernelRangeOp>();
    target.addIllegalOp<VPUASM::ActShaveRtOp>();
    target.addIllegalOp<VPUASM::M2IOp>();
    target.addIllegalOp<VPUASM::NNDMAOp>();

    mlir::RewritePatternSet patterns(&ctx);

    auto mainOps = to_small_vector(netFunc.getOps<ELF::MainOp>());
    VPUX_THROW_UNLESS(mainOps.size() == 1, "Expected exactly one ELF mainOp. Got {0}", mainOps.size());
    auto elfMain = mainOps[0];

    ELF::SymbolReferenceMap symRefMap(elfMain, true);

    patterns.add<NNDMARewriter>(&ctx, _log, symRefMap);
    patterns.add<M2IRewriter>(&ctx, _log, symRefMap);
    patterns.add<ActShaveRtRewriter>(&ctx, _log);
    patterns.add<ActKernelInvocationRewriter>(&ctx, _log, symRefMap);
    patterns.add<ActKernelRangeRewriter>(&ctx, _log, symRefMap);

    if (mlir::failed(mlir::applyPartialConversion(netFunc, target, std::move(patterns)))) {
        signalPassFailure();
    }

    return;
}

}  // namespace

//
// createConvertVPUASM2NPUReg40XXRelocsPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUASM2NPUReg40XXRelocsPass(Logger log, bool enableWLM) {
    return std::make_unique<ConvertVPUASM2NPUReg40XXRelocsPass>(log, enableWLM);
}
