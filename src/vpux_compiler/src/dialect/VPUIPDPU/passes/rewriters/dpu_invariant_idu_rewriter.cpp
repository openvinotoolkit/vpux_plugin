//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/dpu_invariant_block_rewriters.hpp"

using namespace vpux;

namespace {

using namespace VPUIPDPU;

struct IDUConfig {
    struct InActivations {
        bool inSparse = false;
    } inActivations;
    struct Weights {
        mlir::Type wMode;
        std::optional<int64_t> poolWtData;
        bool wtSparse = false;
    } weights;
    struct InputLayerCfg {
        int64_t sparsityPattern = 0;
        bool inputCompressed = false;
    } inputLayerCfg;
    struct StorageElement {
        int64_t seSize = 0;
        std::optional<int64_t> numSEsInZDir;
    } storageElement;
    struct Kernel {
        int64_t kernelX = 1;
        int64_t kernelY = 1;
    } kernel;
    struct Stride {
        int64_t strideX = 1;
        int64_t strideY = 1;
    } stride;
    struct WorkloadCfg {
        VPUIPDPU::IDUWorkloadType workloadType = VPUIPDPU::IDUWorkloadType::CONV;
    } workloadCfg;
    struct DepthWiseCfg {
        bool dw3x3s2OptDisable = false;
        std::optional<int64_t> dwOptOffset;
    } depthWiseCfg;
    struct EltWiseCfg {
        bool eltWiseCfgOp = false;
        int64_t elopScaleA = 1;
        int64_t elopScaleB = 1;
        std::optional<float> fpElopScaleA;
        std::optional<float> fpElopScaleB;
        std::optional<VPUIPDPU::IDUEltwiseType> eltwiseType;
    } eltWiseCfg;
};

struct PPETask {
    std::optional<SmallVector<int64_t>> in1QuantMult;
    std::optional<SmallVector<int64_t>> in2QuantMult;
    std::optional<SmallVector<float>> in1QuantMultFp;
    std::optional<SmallVector<float>> in2QuantMultFp;
    std::optional<VPUIPDPU::IDUEltwiseType> eltwiseType;
};

mlir::FailureOr<PPETask> evalPPETasks(mlir::Region& ppeRegion) {
    PPETask ppeTask;

    for (auto ppeTaskOp : ppeRegion.getOps<VPUASM::PPETaskOp>()) {
        if (ppeTaskOp.getIn1QuantMult().has_value()) {
            auto in1QuantMultArrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(ppeTaskOp.getIn1QuantMult().value());
            if (mlir::isa_and_nonnull<mlir::FloatAttr>(in1QuantMultArrayAttr.getValue()[0])) {
                ppeTask.in1QuantMultFp = parseFPArrayAttr<float>(in1QuantMultArrayAttr);
            } else if (mlir::isa_and_nonnull<mlir::IntegerAttr>(in1QuantMultArrayAttr.getValue()[0])) {
                ppeTask.in1QuantMult = parseIntArrayAttr<int64_t>(in1QuantMultArrayAttr);
            }
        }

        if (ppeTaskOp.getIn2QuantMult().has_value()) {
            auto in2QuantMultArrayAttr = mlir::dyn_cast<mlir::ArrayAttr>(ppeTaskOp.getIn2QuantMult().value());
            if (mlir::isa_and_nonnull<mlir::FloatAttr>(in2QuantMultArrayAttr.getValue()[0])) {
                ppeTask.in2QuantMultFp = parseFPArrayAttr<float>(in2QuantMultArrayAttr);
            } else if (mlir::isa_and_nonnull<mlir::IntegerAttr>(in2QuantMultArrayAttr.getValue()[0])) {
                ppeTask.in2QuantMult = parseIntArrayAttr<int64_t>(in2QuantMultArrayAttr);
            }
        }
    }

    return ppeTask;
}

mlir::LogicalResult verifyInQuantConfig(const Logger& log, mlir::Type inType) {
    SmallVector<uint8_t> inQuantZero;

    if (const auto uniformQuantType = inType.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        inQuantZero.push_back(checked_cast<uint8_t>(uniformQuantType.getZeroPoint()));
    } else if (const auto uniformQuantPerAxisType = inType.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        auto zp = uniformQuantPerAxisType.getZeroPoints();
        inQuantZero.resize(zp.size());
        std::transform(zp.begin(), zp.end(), inQuantZero.begin(), [](int64_t a) {
            return checked_cast<uint8_t>(a);
        });
    } else {
        inQuantZero.push_back(0);
    }

    if (inQuantZero.size() != 1) {
        log.error("Mismatch of size between input quant ZP and quant shift vector:  {0} != 1", inQuantZero.size());
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult getInQuantConfig(const Logger& log, mlir::Type in1Type, mlir::Type in2Type, const PPETask& ppeTask,
                                     SmallVector<int64_t>& in1QuantMult, SmallVector<int64_t>& in1QuantShift,
                                     SmallVector<int64_t>& in2QuantMult, SmallVector<int64_t>& in2QuantShift,
                                     SmallVector<float>& in1QuantMultFp, SmallVector<float>& in2QuantMultFp) {
    if (verifyInQuantConfig(log, in1Type).failed()) {
        return mlir::failure();
    }

    if (verifyInQuantConfig(log, in2Type).failed()) {
        return mlir::failure();
    }

    if (ppeTask.in1QuantMult.has_value()) {
        in1QuantMult = ppeTask.in1QuantMult.value();
    }
    if (ppeTask.in2QuantMult.has_value()) {
        in2QuantMult = ppeTask.in2QuantMult.value();
    }
    if (ppeTask.in1QuantMultFp.has_value()) {
        in1QuantMultFp = ppeTask.in1QuantMultFp.value();
    }
    if (ppeTask.in2QuantMultFp.has_value()) {
        in2QuantMultFp = ppeTask.in2QuantMultFp.value();
    }

    in1QuantShift = in2QuantShift = {0};

    return mlir::success();
}

mlir::LogicalResult configureInActivations(const Logger&, IDUConfig::InActivations& config, bool inSparse) {
    config.inSparse = inSparse;

    return mlir::success();
}

mlir::LogicalResult configureWeights(const Logger& log, IDUConfig::Weights& config, VPUIP::NCETaskType taskType,
                                     mlir::Type inActType, mlir::Type weightsType, bool wtSparse) {
    if (taskType == VPUIP::NCETaskType::MAXPOOL || taskType == VPUIP::NCETaskType::AVEPOOL) {
        config.wMode = DPUInvariantBlockRewriter::getBaseType(inActType);
    } else {
        if (!weightsType) {
            log.error("Missing weights data for DPU task {0}", VPUIP::stringifyNCETaskType(taskType));
            return mlir::failure();
        }
        config.wMode = DPUInvariantBlockRewriter::getBaseType(weightsType);
    }

    if (taskType == VPUIP::NCETaskType::AVEPOOL) {
        if (config.wMode.isInteger(CHAR_BIT * sizeof(uint8_t))) {
            config.poolWtData = 0x0101;  // Two I8/U8 values => 0x0101;
        } else if (config.wMode.isF16()) {
            config.poolWtData = 0x3c00;  // fp16 1
        } else if (config.wMode.isBF16()) {
            config.poolWtData = 0x3f80;  // bf16 1
        } else if (config.wMode.isFloat8E5M2()) {
            config.poolWtData = 0x3c;  // bf8 1
        } else if (config.wMode.isFloat8E4M3FN()) {
            config.poolWtData = 0x38;  // hf8 1
        } else {
            log.error("Input data type not supported for AVEPOOL");
            return mlir::failure();
        }
    }

    config.wtSparse = (taskType == VPUIP::NCETaskType::MAXPOOL) || wtSparse;

    return mlir::success();
}

mlir::LogicalResult configureSparsityPattern(const Logger&, IDUConfig::InputLayerCfg& config,
                                             std::optional<int64_t> spPattern,
                                             std::optional<bool> inChannelsCompression) {
    if (spPattern.has_value()) {
        config.sparsityPattern = spPattern.value();
        if (config.sparsityPattern) {
            config.inputCompressed = inChannelsCompression.value_or(0);
        }
    }

    return mlir::success();
}

mlir::LogicalResult configureStorageElement(const Logger& log, IDUConfig::StorageElement& config,
                                            VPUIP::NCETaskType taskType, const vpux::NDTypeInterface& inActType,
                                            bool inSparsityEnabled, std::optional<int64_t> seSize) {
    if (taskType == VPUIP::NCETaskType::CONV || taskType == VPUIP::NCETaskType::CMCONV ||
        taskType == VPUIP::NCETaskType::ELTWISE) {
        auto seSizeVal = seSize.value_or(0);
        if (inSparsityEnabled && seSizeVal) {
            auto inputZ = inActType.getShape()[Dims4D::Act::C];
            if ((taskType == VPUIP::NCETaskType::ELTWISE) && (seSizeVal != inputZ)) {
                log.warning("Storage_element_size ({0}) for eltwise != Z dim ({1}) ---- not tested", seSizeVal, inputZ);
            }
            config.seSize = seSizeVal;
            if (seSizeVal != 0) {
                auto numSEsInZDir = (inputZ / seSizeVal) - 1;
                if (inputZ % seSizeVal) {
                    ++numSEsInZDir;
                }
                config.numSEsInZDir = numSEsInZDir;
            }
        }
    }

    return mlir::success();
}

mlir::LogicalResult configureKernel(const Logger&, IDUConfig::Kernel& config,
                                    std::optional<mlir::ArrayAttr> kernelSize) {
    if (kernelSize.has_value()) {
        auto kernelSizeArray = parseIntArrayAttr<int64_t>(kernelSize.value());
        config.kernelX = kernelSizeArray[1];
        config.kernelY = kernelSizeArray[0];
    }

    return mlir::success();
}

mlir::LogicalResult configureStride(const Logger&, IDUConfig::Stride& config,
                                    std::optional<mlir::ArrayAttr> kernelStrides) {
    if (kernelStrides.has_value()) {
        auto kernelStridesArray = parseIntArrayAttr<int64_t>(kernelStrides.value());
        config.strideX = kernelStridesArray[1];
        config.strideY = kernelStridesArray[0];
    }

    return mlir::success();
}

mlir::LogicalResult configureWorkload(const Logger& log, IDUConfig::WorkloadCfg& config, VPUIP::NCETaskType taskType,
                                      int64_t kernelX, int64_t kernelY) {
    switch (taskType) {
    case VPUIP::NCETaskType::CONV:
        config.workloadType = VPUIPDPU::IDUWorkloadType::CONV;
        break;
    case VPUIP::NCETaskType::DWCONV:
        config.workloadType = VPUIPDPU::IDUWorkloadType::DWCONV;
        break;
    case VPUIP::NCETaskType::CMCONV:
        // All the above are a subtype of convolution
        config.workloadType = VPUIPDPU::IDUWorkloadType::CONV;
        break;
    case VPUIP::NCETaskType::MAXPOOL:
        config.workloadType = VPUIPDPU::IDUWorkloadType::MAXPOOL;
        break;
    case VPUIP::NCETaskType::AVEPOOL:
        config.workloadType = VPUIPDPU::IDUWorkloadType::AVEPOOL;
        break;
    case VPUIP::NCETaskType::ELTWISE: {
        if (kernelX != 1 || kernelY != 1) {
            log.error("Eltwise only supports 1x1 kernel. Got '{0}' x '{1}'", kernelX, kernelY);
            return mlir::failure();
        }
        config.workloadType = VPUIPDPU::IDUWorkloadType::ELTWISE;
    } break;
    case VPUIP::NCETaskType::IDENTITY:
    case VPUIP::NCETaskType::FCL:
    default:
        log.error("Workload not supported '{0}'", VPUIP::stringifyNCETaskType(taskType));
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult configureDepthWiseCfg(const Logger&, IDUConfig::DepthWiseCfg& config, VPUIP::NCETaskType taskType,
                                          std::optional<bool> smallKernelOptimization) {
    if (smallKernelOptimization.value_or(false)) {
        config.dw3x3s2OptDisable = false;
    } else if (taskType == VPUIP::NCETaskType::DWCONV || taskType == VPUIP::NCETaskType::MAXPOOL) {
        config.dw3x3s2OptDisable = true;
    }

    return mlir::success();
}

mlir::LogicalResult configureEltWiseCfg(const Logger& log, IDUConfig::EltWiseCfg& config, VPUIP::NCETaskType taskType,
                                        mlir::Type inActType, mlir::Type weightsType, const PPETask& ppeTask) {
    if (taskType == VPUIP::NCETaskType::ELTWISE) {
        config.eltWiseCfgOp = true;

        if (ppeTask.eltwiseType.has_value()) {
            config.eltwiseType = ppeTask.eltwiseType.value();
        }

        const auto isInputQuantizationProvided =
                (ppeTask.in1QuantMult.has_value() && ppeTask.in2QuantMult.has_value()) ||
                (ppeTask.in1QuantMultFp.has_value() && ppeTask.in2QuantMultFp.has_value());
        SmallVector<int64_t> in1QuantMult, in2QuantMult;
        SmallVector<float> in1QuantMultFp, in2QuantMultFp;
        SmallVector<int64_t> in1QuantShift, in2QuantShift;
        if (isInputQuantizationProvided) {
            if (getInQuantConfig(log, inActType, weightsType, ppeTask, in1QuantMult, in1QuantShift, in2QuantMult,
                                 in2QuantShift, in1QuantMultFp, in2QuantMultFp)
                        .failed()) {
                return mlir::failure();
            }
        } else {
            SmallVector<uint8_t> in1QuantZero, in2QuantZero;
            if (DPUInvariantBlockRewriter::getQuantConfig(log, inActType, in1QuantMult, in1QuantShift, in1QuantZero)
                        .failed()) {
                return mlir::failure();
            }
            if (DPUInvariantBlockRewriter::getQuantConfig(log, weightsType, in2QuantMult, in2QuantShift, in2QuantZero)
                        .failed()) {
                return mlir::failure();
            }
        }

        if (!in1QuantMult.empty() && !in2QuantMult.empty() && !in1QuantShift.empty() && !in2QuantShift.empty()) {
            if (!in1QuantShift[0] && !in2QuantShift[0]) {
                config.elopScaleA = in1QuantMult[0];
                config.elopScaleB = in2QuantMult[0];
            }
        }

        if (!in1QuantMultFp.empty() && !in2QuantMultFp.empty() && !in1QuantShift.empty() && !in2QuantShift.empty()) {
            config.fpElopScaleA = in1QuantMultFp[0];
            config.fpElopScaleB = in2QuantMultFp[0];
        }

        if (DPUInvariantBlockRewriter::getBaseType(inActType).isa<mlir::Float8E4M3FNType>() ||
            DPUInvariantBlockRewriter::getBaseType(inActType).isa<mlir::Float8E5M2Type>() ||
            DPUInvariantBlockRewriter::getBaseType(weightsType).isa<mlir::Float8E5M2Type>() ||
            DPUInvariantBlockRewriter::getBaseType(weightsType).isa<mlir::Float8E5M2Type>()) {
            if (!config.fpElopScaleA.has_value() && !config.fpElopScaleB.has_value()) {
                config.fpElopScaleA = 1;
                config.fpElopScaleB = 1;
            }
        }
    }

    return mlir::success();
}

mlir::LogicalResult configureIDU(const Logger& log, IDUConfig& config, const vpux::NDTypeInterface& inActType,
                                 mlir::Type weightsElementType, VPUIP::NCETaskType taskType,
                                 std::optional<int64_t> spPattern, std::optional<bool> inChannelsCompression,
                                 std::optional<bool> smallKernelOptimization, bool inActSparse, bool weightsSparse,
                                 std::optional<mlir::ArrayAttr> kernelSize,
                                 std::optional<mlir::ArrayAttr> kernelStrides, std::optional<int64_t> seSize,
                                 const PPETask& ppeTask) {
    // IDUInActivations
    if (configureInActivations(log, config.inActivations, inActSparse).failed()) {
        return mlir::failure();
    }

    // IDUWeights
    auto inActElementType = inActType.cast<mlir::MemRefType>().getElementType();
    if (configureWeights(log, config.weights, taskType, inActElementType, weightsElementType, weightsSparse).failed()) {
        return mlir::failure();
    }

    // IDUInputLayerCfg
    if (configureSparsityPattern(log, config.inputLayerCfg, spPattern, inChannelsCompression).failed()) {
        return mlir::failure();
    }

    // IDUStorageElement
    if (configureStorageElement(log, config.storageElement, taskType, inActType, inActSparse, seSize).failed()) {
        return mlir::failure();
    }

    // IDUKernel
    if (configureKernel(log, config.kernel, kernelSize).failed()) {
        return mlir::failure();
    }

    // IDUStride
    if (configureStride(log, config.stride, kernelStrides).failed()) {
        return mlir::failure();
    }

    // IDUWorkloadCfg
    if (configureWorkload(log, config.workloadCfg, taskType, config.kernel.kernelX, config.kernel.kernelY).failed()) {
        return mlir::failure();
    }

    // IDUDepthWiseCfg
    if (configureDepthWiseCfg(log, config.depthWiseCfg, taskType, smallKernelOptimization).failed()) {
        return mlir::failure();
    }

    // IDUEltWiseCfg
    if (configureEltWiseCfg(log, config.eltWiseCfg, taskType, inActElementType, weightsElementType, ppeTask).failed()) {
        return mlir::failure();
    }

    return mlir::success();
}

mlir::IntegerAttr getI64IntegerAttrOrNull(mlir::OpBuilder& builder, const std::optional<int64_t>& attr) {
    if (attr.has_value()) {
        return builder.getI64IntegerAttr(attr.value());
    }

    return nullptr;
}

mlir::LogicalResult buildIDUConfig(mlir::OpBuilder& builder, const mlir::Location& loc, const IDUConfig& config,
                                   mlir::Value inAct) {
    // IDUInActivations
    builder.create<VPUIPDPU::IDUInActivationsOp>(loc, inAct, config.inActivations.inSparse);

    // IDUWeights
    auto poolWtDataAttr = getI64IntegerAttrOrNull(builder, config.weights.poolWtData);
    builder.create<VPUIPDPU::IDUWeightsOp>(loc, config.weights.wMode, poolWtDataAttr, config.weights.wtSparse);

    // IDUInputLayerCfg
    if (config.inputLayerCfg.sparsityPattern) {
        builder.create<VPUIPDPU::IDUInputLayerCfgOp>(loc, config.inputLayerCfg.sparsityPattern,
                                                     config.inputLayerCfg.inputCompressed);
    }

    // IDUStorageElement
    if (config.storageElement.seSize) {
        auto numSEsInZDirAttr = getI64IntegerAttrOrNull(builder, config.storageElement.numSEsInZDir);
        builder.create<VPUIPDPU::IDUStorageElementOp>(loc, config.storageElement.seSize, numSEsInZDirAttr);
    }

    // IDUKernel
    builder.create<VPUIPDPU::IDUKernelOp>(loc, config.kernel.kernelX, config.kernel.kernelY);

    // IDUStride
    builder.create<VPUIPDPU::IDUStrideOp>(loc, config.stride.strideX, config.stride.strideY);

    // IDUWorkloadCfg
    builder.create<VPUIPDPU::IDUWorkloadCfgOp>(loc, config.workloadCfg.workloadType);

    // IDUDepthWiseCfg
    if (config.depthWiseCfg.dw3x3s2OptDisable || config.depthWiseCfg.dwOptOffset.has_value()) {
        auto dwOptOffsetAttr = getI64IntegerAttrOrNull(builder, config.depthWiseCfg.dwOptOffset);
        builder.create<VPUIPDPU::IDUDepthWiseCfgOp>(loc, config.depthWiseCfg.dw3x3s2OptDisable, dwOptOffsetAttr);
    }

    // IDUEltWiseCfg
    if (config.eltWiseCfg.eltWiseCfgOp) {
        if (config.eltWiseCfg.fpElopScaleA.has_value() && config.eltWiseCfg.fpElopScaleB.has_value()) {
            auto fpElopScaleAAttr = builder.getF32FloatAttr(config.eltWiseCfg.fpElopScaleA.value());
            auto fpElopScaleBAttr = builder.getF32FloatAttr(config.eltWiseCfg.fpElopScaleB.value());
            builder.create<VPUIPDPU::IDUEltWiseCfgOp>(loc, fpElopScaleAAttr, fpElopScaleBAttr);
        } else {
            auto elopScaleAAttr = getI64IntegerAttrOrNull(builder, config.eltWiseCfg.elopScaleA);
            auto elopScaleBAttr = getI64IntegerAttrOrNull(builder, config.eltWiseCfg.elopScaleB);
            builder.create<VPUIPDPU::IDUEltWiseCfgOp>(loc, elopScaleAAttr, elopScaleBAttr);
        }
    }

    return mlir::success();
}  // namespace

}  // namespace

namespace vpux {
namespace VPUIPDPU {

DPUInvariantIDURewriter::DPUInvariantIDURewriter(VPUASM::DPUInvariantOp origInvOp, mlir::Block* invBlock,
                                                 std::map<BlockArg, size_t>& invBlockArgsPos,
                                                 mlir::PatternRewriter& rewriter, const Logger& log)
        : DPUInvariantBlockRewriter(origInvOp, invBlock, invBlockArgsPos, rewriter, log) {
}

mlir::LogicalResult DPUInvariantIDURewriter::rewrite() {
    if (insertEntryBlock<VPUIPDPU::IDUCfgOp>().failed()) {
        return mlir::failure();
    }

    IDUConfig config;
    auto inAct = getInvBlockArg(DPUInvariantBlockRewriter::BlockArg::ACT_IN);
    mlir::Type weightsType;
    if (auto weights = getInvBlockArg(DPUInvariantBlockRewriter::BlockArg::WEIGHTS)) {
        weightsType = weights.getType().cast<mlir::MemRefType>().getElementType();
    }

    auto ppeTask = evalPPETasks(_origInvOp.getPpe());
    if (mlir::failed(ppeTask)) {
        return mlir::failure();
    }

    if (configureIDU(
                _log, config, inAct.getType(), weightsType, _origInvOp.getNceTaskType(), _origInvOp.getCmSpPattern(),
                _origInvOp.getInputChannelsCompression(), _origInvOp.getIsSmallKernelOptimized(),
                getInvBlockArg(DPUInvariantBlockRewriter::BlockArg::ACT_SPARSE_MAP_IN) != nullptr,
                getInvBlockArg(DPUInvariantBlockRewriter::BlockArg::WEIGHTS_SPARSE_MAP) != nullptr,
                _origInvOp.getKernelSize(), _origInvOp.getKernelStrides(), _origInvOp.getInputSeSize(), ppeTask.value())
                .failed()) {
        return mlir::failure();
    }

    if (buildIDUConfig(_rewriter, _origInvOp.getLoc(), config, inAct).failed()) {
        return mlir::failure();
    }

    return mlir::success();
}

}  // namespace VPUIPDPU
}  // namespace vpux
