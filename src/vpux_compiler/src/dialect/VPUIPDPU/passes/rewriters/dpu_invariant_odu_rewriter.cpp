//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/dpu_invariant_block_rewriters.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"

using namespace vpux;

namespace {

struct ODUConfig {
    struct OutTensorSize {
        uint32_t dimX = 0;
        uint32_t dimY = 0;
        uint32_t dimZ = 0;
    } outTensorSize;
    struct DataReuse {
        VPUIPDPU::ODUActivationReuseMode activationReuse = VPUIPDPU::ODUActivationReuseMode::NTHW_1;
    } dataReuse;
    struct PermuteData {
        VPUIPDPU::ODUPermuteDataMode permuteMode = VPUIPDPU::ODUPermuteDataMode::PERMUTE_ZXY;
    } permuteData;
    struct Sparsity {
        std::optional<bool> compressionEnabled;
        std::optional<int64_t> sparseValue;
    } sparsity;
    struct SwizzleData {
        VPUIPDPU::DPUSwizzleKey swizzleKey = VPUIPDPU::DPUSwizzleKey::SWIZZLE_OFF;
    } swizzleData;
    struct OutActivations {
        std::optional<VPUIPDPU::ODUDataBitWidth> dataWidth;
    } outActivations;
    struct MemoryMode {
        VPUIPDPU::ODUMemoryMode memMode = VPUIPDPU::ODUMemoryMode::MODE_DENSE;
    } memoryMode;
};

std::optional<VPUIPDPU::ODUDataBitWidth> getOutDataWidth(mlir::Type outDataType) {
    std::optional<VPUIPDPU::ODUDataBitWidth> outDataWidth;

    if (outDataType.isF32()) {
        return VPUIPDPU::ODUDataBitWidth::ODU_DTYPE_32BIT;
    } else if (outDataType.isF16()) {
        return VPUIPDPU::ODUDataBitWidth::ODU_DTYPE_16BIT;
    } else if (outDataType.isBF16()) {
        return VPUIPDPU::ODUDataBitWidth::ODU_DTYPE_16BIT;
    } else if (outDataType.isFloat8E4M3FN()) {
        return VPUIPDPU::ODUDataBitWidth::ODU_DTYPE_8BIT;
    } else if (outDataType.isFloat8E5M2()) {
        return VPUIPDPU::ODUDataBitWidth::ODU_DTYPE_8BIT;
    } else if (outDataType.isSignedInteger(CHAR_BIT * sizeof(int32_t))) {
        return VPUIPDPU::ODUDataBitWidth::ODU_DTYPE_32BIT;
    } else if (outDataType.isSignedInteger(CHAR_BIT * sizeof(int8_t))) {
        return VPUIPDPU::ODUDataBitWidth::ODU_DTYPE_8BIT;
    } else if (outDataType.isSignedInteger(4)) {
        return VPUIPDPU::ODUDataBitWidth::ODU_DTYPE_4BIT;
    } else if (outDataType.isInteger(CHAR_BIT * sizeof(uint8_t))) {
        return VPUIPDPU::ODUDataBitWidth::ODU_DTYPE_8BIT;
    } else if (outDataType.isInteger(4)) {
        return VPUIPDPU::ODUDataBitWidth::ODU_DTYPE_4BIT;
    } else if (outDataType.isInteger(2)) {
        return VPUIPDPU::ODUDataBitWidth::ODU_DTYPE_2BIT;
    } else if (outDataType.isInteger(1)) {
        return VPUIPDPU::ODUDataBitWidth::ODU_DTYPE_1BIT;
    } else if (outDataType.isa<mlir::quant::QuantizedType>()) {
        return getOutDataWidth(outDataType.cast<mlir::quant::QuantizedType>().getStorageType());
    }

    return outDataWidth;
}

uint8_t getQuantZeroPoint(mlir::Type type) {
    uint8_t quantZeroPoint = 0;

    if (const auto qType = type.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        quantZeroPoint = checked_cast<uint8_t>(qType.getZeroPoint());
    } else if (const auto qPerAxisType = type.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        auto qtypeQuantZp = qPerAxisType.getZeroPoints();
        quantZeroPoint = checked_cast<uint8_t>(qtypeQuantZp[0]);
    }

    return quantZeroPoint;
}

mlir::LogicalResult configureOutTensorSize(const Logger& log, ODUConfig::OutTensorSize& config,
                                           VPUIPDPU::ODUPermuteDataMode permuteMode, const Strides& outStrides) {
    auto outStridesVec = getFloatStrides(outStrides);
    if (std::count(outStridesVec.begin(), outStridesVec.end(), 0.0f)) {
        log.error("One or more activation output strides are zero, invalid configuration");
        return mlir::failure();
    }

    auto strideB = outStridesVec[Dims4D::Act::N.ind()];
    auto strideX = outStridesVec[Dims4D::Act::W.ind()];
    auto strideY = outStridesVec[Dims4D::Act::H.ind()];
    auto strideZ = outStridesVec[Dims4D::Act::C.ind()];

    switch (permuteMode) {
    case VPUIPDPU::ODUPermuteDataMode::PERMUTE_ZXY:
        // NHWC
        config.dimY = strideB / strideY;
        config.dimX = strideY / strideX;
        config.dimZ = strideX / strideZ;
        break;
    case VPUIPDPU::ODUPermuteDataMode::PERMUTE_YZX:
        // NWCH
        config.dimX = strideB / strideX;
        config.dimZ = strideX / strideZ;
        config.dimY = strideZ / strideY;
        break;
    case VPUIPDPU::ODUPermuteDataMode::PERMUTE_ZYX:
        // NWHC
        config.dimX = strideB / strideX;
        config.dimY = strideX / strideY;
        config.dimZ = strideY / strideZ;
        break;
    case VPUIPDPU::ODUPermuteDataMode::PERMUTE_XZY:
        // NHCW
        config.dimY = strideB / strideY;
        config.dimZ = strideY / strideZ;
        config.dimX = strideZ / strideX;
        break;
    case VPUIPDPU::ODUPermuteDataMode::PERMUTE_YXZ:
        // NCWH
        config.dimZ = strideB / strideZ;
        config.dimX = strideZ / strideX;
        config.dimY = strideX / strideY;
        break;
    case VPUIPDPU::ODUPermuteDataMode::PERMUTE_XYZ:
        // NCHW
        config.dimZ = strideB / strideZ;
        config.dimY = strideZ / strideY;
        config.dimX = strideY / strideX;
        break;
    default:
        log.error("Wrong permutation {0}, invalid configuration", stringifyODUPermuteDataMode(permuteMode));
        return mlir::failure();
    }

    if (!(config.dimX && config.dimY && config.dimZ)) {
        log.error("All output dimensions must be >= 1: dimX={0}, dimY={1}, dimZ={2}", config.dimX, config.dimY,
                  config.dimZ);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult configureDataReuse(const Logger& log, ODUConfig::DataReuse& config, VPU::MPEMode mpeFrequentMode,
                                       VPUIP::NCETaskType dpuTaskType) {
    // Hardware only supports MPE_Mode_CUBOID_8x16 for Elementwise addition
    if (dpuTaskType == VPUIP::NCETaskType::ELTWISE) {
        mpeFrequentMode = VPU::MPEMode::CUBOID_8x16;
    }

    switch (mpeFrequentMode) {
    case VPU::MPEMode::CUBOID_4x16:  // NTH = 1, NTW=4, NTK = 16 (4, 16)
        config.activationReuse = VPUIPDPU::ODUActivationReuseMode::NTHW_4;
        break;
    case VPU::MPEMode::CUBOID_8x16:  // NTH = 2, NTW=4, NTK = 8 (8, 8)
    case VPU::MPEMode::VECTOR:       // dpu_runtime::MPE_GRID_16x1 not valid for NPU37XX
        config.activationReuse = VPUIPDPU::ODUActivationReuseMode::NTHW_8;
        break;
    case VPU::MPEMode::CUBOID_16x16:  // NTH = 4, NTW=4, NTK = 4  (16, 4)
        config.activationReuse = VPUIPDPU::ODUActivationReuseMode::NTHW_16;
        break;
    default:
        log.error("ODU NTHW mode not supported for MPE mode {}", mpeFrequentMode);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult configurePermuteMode(const Logger& log, ODUConfig::PermuteData& config,
                                         const DimsOrder& outDimsOrder) {
    if (outDimsOrder == DimsOrder::NHWC) {
        config.permuteMode = VPUIPDPU::ODUPermuteDataMode::PERMUTE_ZXY;
    } else if (outDimsOrder == DimsOrder::NWHC) {
        config.permuteMode = VPUIPDPU::ODUPermuteDataMode::PERMUTE_ZYX;
    } else if (outDimsOrder == DimsOrder::NWCH) {
        config.permuteMode = VPUIPDPU::ODUPermuteDataMode::PERMUTE_YZX;
    } else if (outDimsOrder == DimsOrder::NCWH) {
        config.permuteMode = VPUIPDPU::ODUPermuteDataMode::PERMUTE_YXZ;
    } else if (outDimsOrder == DimsOrder::NHCW) {
        config.permuteMode = VPUIPDPU::ODUPermuteDataMode::PERMUTE_XZY;
    } else if (outDimsOrder == DimsOrder::NCHW) {
        config.permuteMode = VPUIPDPU::ODUPermuteDataMode::PERMUTE_XYZ;
    } else {
        log.error("Can't get ODU permutation by output dimsOrder: '{0}'", outDimsOrder);
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult configureSparsity(const Logger&, ODUConfig::Sparsity& config, bool outSparsityEnabled,
                                      uint8_t sparseValue) {
    if (outSparsityEnabled) {
        config.compressionEnabled = true;
        config.sparseValue = sparseValue;
    }

    return mlir::success();
}

mlir::LogicalResult configureSwizzleData(const Logger& log, ODUConfig::SwizzleData& config,
                                         std::optional<int64_t> outSwizzling) {
    if (outSwizzling.has_value()) {
        auto swizzleKey = VPUIPDPU::symbolizeDPUSwizzleKey(outSwizzling.value());
        if (!swizzleKey.has_value()) {
            log.error("Invalid output swizzle key '{0}'", outSwizzling.value());
            return mlir::failure();
        }
        config.swizzleKey = swizzleKey.value();
    }

    return mlir::success();
}

mlir::LogicalResult configureOutActivations(const Logger& log, ODUConfig::OutActivations& config,
                                            mlir::Type outDataType) {
    auto outDataWidth = getOutDataWidth(outDataType);

    if (!outDataWidth.has_value()) {
        log.error("Invalid output data type '{0}'", outDataType);
        return mlir::failure();
    }

    if (outDataType.isa<mlir::quant::QuantizedType>()) {
        config.dataWidth = outDataWidth.value();
    }

    return mlir::success();
}

mlir::LogicalResult configureMemoryMode(const Logger& log, ODUConfig::MemoryMode& config,
                                        std::optional<bool> isSuperdense) {
    if (isSuperdense.has_value()) {
        auto memMode = VPUIPDPU::symbolizeODUMemoryMode(isSuperdense.value());
        if (!memMode.has_value()) {
            log.error("Invalid output mode '{0}'", isSuperdense.value());
            return mlir::failure();
        }
        config.memMode = memMode.value();
    }

    return mlir::success();
}

mlir::LogicalResult configureODU(const Logger& log, ODUConfig& config, const NDTypeInterface& outActType,
                                 VPU::MPEMode mpeFrequentMode, VPUIP::NCETaskType dpuTaskType,
                                 std::optional<int64_t> outSwizzling, std::optional<bool> isSuperdense,
                                 bool outSparsityEnabled) {
    if (configurePermuteMode(log, config.permuteData, outActType.getDimsOrder()).failed()) {
        return mlir::failure();
    }

    if (configureOutTensorSize(log, config.outTensorSize, config.permuteData.permuteMode, outActType.getStrides())
                .failed()) {
        return mlir::failure();
    }

    if (configureDataReuse(log, config.dataReuse, mpeFrequentMode, dpuTaskType).failed()) {
        return mlir::failure();
    }

    if (configureSparsity(log, config.sparsity, outSparsityEnabled, getQuantZeroPoint(outActType.getElementType()))
                .failed()) {
        return mlir::failure();
    }

    if (configureSwizzleData(log, config.swizzleData, outSwizzling).failed()) {
        return mlir::failure();
    }

    if (configureOutActivations(log, config.outActivations, outActType.getElementType()).failed()) {
        return mlir::failure();
    }

    if (configureMemoryMode(log, config.memoryMode, isSuperdense).failed()) {
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult buildODUConfig(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                   const ODUConfig& config, mlir::Value outAct, mlir::Value outSparsityMap) {
    // ODUOutTensorSize
    builder.create<VPUIPDPU::ODUOutTensorSizeOp>(loc, config.outTensorSize.dimX, config.outTensorSize.dimY,
                                                 config.outTensorSize.dimZ);

    // ODUDataReuse
    if (config.dataReuse.activationReuse != VPUIPDPU::ODUActivationReuseMode::NTHW_1) {
        builder.create<VPUIPDPU::ODUDataReuseOp>(loc, config.dataReuse.activationReuse);
    }

    // ODUPermuteData
    if (config.permuteData.permuteMode != VPUIPDPU::ODUPermuteDataMode::PERMUTE_ZXY) {
        builder.create<VPUIPDPU::ODUPermuteDataOp>(loc, config.permuteData.permuteMode);
    }

    // ODUSparsity
    if (config.sparsity.compressionEnabled.has_value()) {
        if (!outSparsityMap) {
            log.error("Expected output_sparsity_map operand in ODU invariant");
            return mlir::failure();
        }
        mlir::IntegerAttr sparseValue = nullptr;
        if (config.sparsity.sparseValue.has_value() && config.sparsity.sparseValue.value()) {
            sparseValue = builder.getI64IntegerAttr(config.sparsity.sparseValue.value());
        }
        builder.create<VPUIPDPU::ODUSparsityOp>(loc, outSparsityMap, nullptr, sparseValue);
    }

    // ODUSwizzleData
    if (config.swizzleData.swizzleKey != VPUIPDPU::DPUSwizzleKey::SWIZZLE_OFF) {
        builder.create<VPUIPDPU::ODUSwizzleDataOp>(loc, config.swizzleData.swizzleKey);
    }

    // ODUOutActivations
    if (config.outActivations.dataWidth.has_value()) {
        auto dataWidthAttr =
                VPUIPDPU::ODUDataBitWidthAttr::get(builder.getContext(), config.outActivations.dataWidth.value());
        builder.create<VPUIPDPU::ODUOutActivationsOp>(loc, outAct, nullptr, dataWidthAttr);
    } else {
        builder.create<VPUIPDPU::ODUOutActivationsOp>(loc, outAct, nullptr, nullptr);
    }

    // ODUMemoryMode
    if (config.memoryMode.memMode != VPUIPDPU::ODUMemoryMode::MODE_DENSE) {
        builder.create<VPUIPDPU::ODUMemoryModeOp>(loc, config.memoryMode.memMode);
    }

    // ODUCmxPorts
    // invariant.odu_cfg.odu_cfg_bf.cmx_port_muxing_disable not set in nce_lib
    // leaving field value to reset initialization in lowering VPUASM2NPUReg40XX

    // ODUWriteCombineBuffer
    // invariant.odu_cfg.odu_cfg_bf.wcb_ac_mode/wcb_sp_mode not set in nce_lib
    // leaving field values to reset initialization in lowering VPUASM2NPUReg40XX

    return mlir::success();
}

}  // namespace

namespace vpux {
namespace VPUIPDPU {

DPUInvariantODURewriter::DPUInvariantODURewriter(VPUASM::DPUInvariantOp origInvOp, mlir::Block* invBlock,
                                                 std::map<BlockArg, size_t>& invBlockArgsPos,
                                                 mlir::PatternRewriter& rewriter, const Logger& log)
        : DPUInvariantBlockRewriter(origInvOp, invBlock, invBlockArgsPos, rewriter, log) {
}

mlir::LogicalResult DPUInvariantODURewriter::rewrite(ELF::SymbolReferenceMap& symRefMap) {
    if (insertEntryBlock<VPUIPDPU::ODUCfgOp>().failed()) {
        return mlir::failure();
    }

    ODUConfig config;
    mlir::MemRefType outType;
    uint64_t outputSwizzling = 0;
    if (!_origInvOp.getIsContinued() && _origInvOp.getOutput()) {
        auto outBuffer = symRefMap.lookupSymbol(_origInvOp.getOutput().value());
        outType = getBufferType(outBuffer);
        outputSwizzling = getSwizzlingKey(outBuffer);
    } else if (_origInvOp.getIsContinued() && _origInvOp.getOutputTypeContinued()) {
        auto outBufferType = _origInvOp.getOutputTypeContinued().value();
        outType = outBufferType.getMemref();
        outputSwizzling = outBufferType.getTraits().getSwizzlingKey();
    } else {
        _log.error("Expected either output buffer or output type for continued mode");
        return mlir::failure();
    }

    if (configureODU(_log, config, outType, _origInvOp.getMpeFrequentMode(), _origInvOp.getNceTaskType(),
                     outputSwizzling, _origInvOp.getIsSuperdense(), _origInvOp.getOutputSparsityMap().has_value())
                .failed()) {
        return mlir::failure();
    }

    if (buildODUConfig(_rewriter, _origInvOp.getLoc(), _log, config,
                       getInvBlockArg(DPUInvariantBlockRewriter::BlockArg::ACT_OUT),
                       getInvBlockArg(DPUInvariantBlockRewriter::BlockArg::ACT_SPARSE_MAP_OUT))
                .failed()) {
        return mlir::failure();
    }

    return mlir::success();
}

}  // namespace VPUIPDPU
}  // namespace vpux
