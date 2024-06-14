//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/dpu_variant_block_rewriters.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"

namespace {

using namespace vpux;

int64_t size(int64_t start, int64_t end) {
    return end - start + 1;
}

mlir::LogicalResult buildIDUWorkloadSet(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger&,
                                        VPUIP::NCETaskType taskType, const SmallVector<int64_t>& inStart,
                                        const SmallVector<int64_t>& inEnd, int64_t outStartZ, int64_t outEndZ,
                                        const vpux::NDTypeInterface& inActType) {
    auto startX = inStart[0];
    auto startY = inStart[1];
    int64_t startZ = 0;
    auto sizeX = size(startX, inEnd[0]);
    auto sizeY = size(startY, inEnd[1]);
    int64_t sizeZ = 0;

    if (taskType == VPUIP::NCETaskType::DWCONV || taskType == VPUIP::NCETaskType::MAXPOOL ||
        taskType == VPUIP::NCETaskType::AVEPOOL) {
        startZ = outStartZ;
        sizeZ = size(outStartZ, outEndZ);
    } else {
        auto inputZ = inActType.getShape()[Dims4D::Act::C];
        if (inputZ < 16) {
            sizeZ = 16;
        } else {
            sizeZ = inputZ;
        }
    }

    builder.create<VPUIPDPU::IDUWorkloadSetOp>(loc, startX, startY, startZ, sizeX, sizeY, sizeZ);

    return mlir::success();
}

mlir::LogicalResult buildIDUWeightSet(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                      int64_t outStartZ, int64_t outEndZ, std::optional<int64_t> outChannelOffset,
                                      VPUIP::NCETaskType taskType, const vpux::NDTypeInterface& inActType,
                                      const vpux::NDTypeInterface& outActType, const vpux::NDTypeInterface& weightsType,
                                      std::optional<mlir::ArrayAttr> kernelSize) {
    // weight_start is updated during run-time relocation by addition with weight_table address
    auto outputZ = outActType.getShape()[Dims4D::Act::C];
    auto weightStart = (outStartZ - outChannelOffset.value_or(0)) % outputZ;
    weightStart <<= 4;

    auto inputZ = inActType.getShape()[Dims4D::Act::C];
    auto outSizeZ = size(outStartZ, outEndZ);
    auto weightNum = outSizeZ;
    int64_t weightSize = 0;
    int64_t kernelX = 1, kernelY = 1;

    if (kernelSize.has_value()) {
        auto kernelSizeArray = parseIntArrayAttr<int64_t>(kernelSize.value());
        kernelX = kernelSizeArray[1];
        kernelY = kernelSizeArray[0];
    }

    switch (taskType) {
    case VPUIP::NCETaskType::CONV:
    case VPUIP::NCETaskType::CMCONV: {
        weightSize = kernelX * kernelY;
        if (inActType.getShape()[Dims4D::Act::C] < 16) {
            if (!weightsType) {
                log.error("Missing weights for DPU task type {0}", VPUIP::stringifyNCETaskType(taskType));
                return mlir::failure();
            }
            weightNum = weightsType.getShape()[Dims4D::Act::N];
            weightSize *= 16;
        } else {
            weightSize *= inputZ;
        }
    } break;
    case VPUIP::NCETaskType::DWCONV:
    case VPUIP::NCETaskType::AVEPOOL:
    case VPUIP::NCETaskType::MAXPOOL:
        weightSize = kernelX * kernelY * outSizeZ;
        break;
    case VPUIP::NCETaskType::ELTWISE: {
        auto inputX = inActType.getShape()[Dims4D::Act::W], inputY = inActType.getShape()[Dims4D::Act::H];
        weightSize = inputX * inputY * inputZ;
    } break;
    default:
        break;
    }

    builder.create<VPUIPDPU::IDUWeightSetOp>(loc, weightStart, weightNum, weightSize);

    return mlir::success();
}

mlir::LogicalResult buildIDUPadding(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger&,
                                    VPU::PaddingAttr pad) {
    builder.create<VPUIPDPU::IDUPaddingOp>(loc, pad);

    return mlir::success();
}

mlir::LogicalResult buildIDUActSwizzle(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                       std::optional<int64_t> inSwizzling) {
    if (inSwizzling.has_value()) {
        auto swizzleKey = VPUIPDPU::symbolizeDPUSwizzleKey(inSwizzling.value());
        if (!swizzleKey.has_value()) {
            log.error("Invalid input swizzle key '{0}'", inSwizzling.value());
            return mlir::failure();
        }
        if (swizzleKey.value() != VPUIPDPU::DPUSwizzleKey::SWIZZLE_OFF) {
            builder.create<VPUIPDPU::IDUActSwizzleOp>(loc, swizzleKey.value());
        }
    }

    return mlir::success();
}

mlir::LogicalResult buildIDUWeightSwizzle(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                          std::optional<int64_t> weightsSwizzling) {
    if (weightsSwizzling.has_value()) {
        auto swizzleKey = VPUIPDPU::symbolizeDPUSwizzleKey(weightsSwizzling.value());
        if (!swizzleKey.has_value()) {
            log.error("Invalid weights swizzle key '{0}'", weightsSwizzling.value());
            return mlir::failure();
        }
        if (swizzleKey.value() != VPUIPDPU::DPUSwizzleKey::SWIZZLE_OFF) {
            builder.create<VPUIPDPU::IDUWeightSwizzleOp>(loc, swizzleKey.value());
        }
    }

    return mlir::success();
}

mlir::LogicalResult buildIDUNthwNtk(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                    VPU::MPEMode mpeFrequentMode, VPUIP::NCETaskType dpuTaskType) {
    // Hardware only supports MPE_Mode_CUBOID_8x16 for Elementwise addition
    if (dpuTaskType == VPUIP::NCETaskType::ELTWISE) {
        mpeFrequentMode = VPU::MPEMode::CUBOID_8x16;
    }

    auto activationReuse = VPUIPDPU::IDUNthwNtk::NTHW_NTK_8_8;

    switch (mpeFrequentMode) {
    case VPU::MPEMode::CUBOID_4x16:  // NTH = 1, NTW=4, NTK = 16 (4, 16)
        activationReuse = VPUIPDPU::IDUNthwNtk::NTHW_NTK_4_16;
        break;
    case VPU::MPEMode::CUBOID_8x16:  // NTH = 2, NTW=4, NTK = 8 (8, 8)
    case VPU::MPEMode::VECTOR:
        activationReuse = VPUIPDPU::IDUNthwNtk::NTHW_NTK_8_8;
        break;
    case VPU::MPEMode::CUBOID_16x16:  // NTH = 4, NTW=4, NTK = 4  (16, 4)
        activationReuse = VPUIPDPU::IDUNthwNtk::NTHW_NTK_16_4;
        break;
    default:
        log.error("ODU NTHW mode not supported for MPE mode {}", mpeFrequentMode);
        return mlir::failure();
    }

    if (activationReuse != VPUIPDPU::IDUNthwNtk::NTHW_NTK_8_8) {
        builder.create<VPUIPDPU::IDUNthwNtkOp>(loc, activationReuse);
    }

    return mlir::success();
}

mlir::LogicalResult buildIDUSEDense(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger&, bool seDense) {
    if (seDense) {
        builder.create<VPUIPDPU::IDUSEDenseOp>(loc);
    }

    return mlir::success();
}

mlir::LogicalResult buildIDUConvContinue(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger&,
                                         std::optional<bool> isContinued) {
    if (isContinued.value_or(false)) {
        builder.create<VPUIPDPU::IDUConvContinueOp>(loc);
    }

    return mlir::success();
}

}  // namespace

namespace vpux {
namespace VPUIPDPU {

DPUVariantIDURewriter::DPUVariantIDURewriter(VPUASM::DPUVariantOp origVarOp, mlir::PatternRewriter& rewriter,
                                             const Logger& log)
        : DPUVariantBlockRewriter(origVarOp, rewriter, log) {
}

mlir::LogicalResult DPUVariantIDURewriter::rewrite(ELF::SymbolReferenceMap& symRefMap) {
    auto origInvOp = mlir::cast<VPUASM::DPUInvariantOp>(symRefMap.lookupSymbol(_origVarOp.getInvariant()));

    auto inAct = symRefMap.lookupSymbol(origInvOp.getInput());
    auto inActType = getBufferType(inAct);
    auto inSwizzlingKey = getSwizzlingKey(inAct);

    mlir::MemRefType outActType;
    if (!origInvOp.getIsContinued() && origInvOp.getOutput()) {
        auto outBuffer = symRefMap.lookupSymbol(origInvOp.getOutput().value());
        outActType = getBufferType(outBuffer);
    } else if (origInvOp.getIsContinued() && origInvOp.getOutputTypeContinued()) {
        auto outBufferType = origInvOp.getOutputTypeContinued().value();
        outActType = outBufferType.getMemref();
    } else {
        _log.error("Expected either output buffer or output type for continued mode");
        return mlir::failure();
    }

    mlir::Type weightsType;
    std::optional<int64_t> weightsSwizzlingKey;
    if (origInvOp.getWeights()) {
        auto weights = symRefMap.lookupSymbol(origInvOp.getWeights().value());
        weightsType = getBufferType(weights);
        weightsSwizzlingKey = getSwizzlingKey(weights);
    }

    auto inputStarts = parseIntArrayAttr<int64_t>(_origVarOp.getInStart());
    auto inputEnds = parseIntArrayAttr<int64_t>(_origVarOp.getInEnd());
    auto outStartZ = parseIntArrayAttr<int64_t>(_origVarOp.getStart())[2];
    auto outEndZ = parseIntArrayAttr<int64_t>(_origVarOp.getEnd())[2];
    // IDUWorkloadSet
    if (VPU::getArch(_origVarOp) == VPU::ArchKind::NPU40XX) {
        _rewriter.create<VPUIPDPU::IDUWorkloadSetOp>(
                _origVarOp.getLoc(), inputStarts[0], inputStarts[1], inputStarts[2], size(inputStarts[0], inputEnds[0]),
                size(inputStarts[1], inputEnds[1]), size(inputStarts[2], inputEnds[2]));
    } else {
        if (buildIDUWorkloadSet(_rewriter, _origVarOp.getLoc(), _log, origInvOp.getNceTaskType(), inputStarts,
                                inputEnds, outStartZ, outEndZ, inActType)
                    .failed()) {
            return mlir::failure();
        }
    }

    // IDUWeightSet
    if (buildIDUWeightSet(_rewriter, _origVarOp.getLoc(), _log, outStartZ, outEndZ, origInvOp.getOutChannelOffset(),
                          origInvOp.getNceTaskType(), inActType, outActType, weightsType, origInvOp.getKernelSize())
                .failed()) {
        return mlir::failure();
    }

    // IDUPadding
    if (buildIDUPadding(_rewriter, _origVarOp.getLoc(), _log, _origVarOp.getPad()).failed()) {
        return mlir::failure();
    }

    // IDUActSwizzle
    if (buildIDUActSwizzle(_rewriter, _origVarOp.getLoc(), _log, inSwizzlingKey).failed()) {
        return mlir::failure();
    }

    // IDUWeightSwizzle
    if (buildIDUWeightSwizzle(_rewriter, _origVarOp.getLoc(), _log, weightsSwizzlingKey).failed()) {
        return mlir::failure();
    }

    // IDUNthwNtk
    if (buildIDUNthwNtk(_rewriter, _origVarOp.getLoc(), _log, origInvOp.getMpeFrequentMode(),
                        origInvOp.getNceTaskType())
                .failed()) {
        return mlir::failure();
    }

    // IDUSEDense
    if (buildIDUSEDense(_rewriter, _origVarOp.getLoc(), _log, !origInvOp.getInputStorageElementTable().has_value())
                .failed()) {
        return mlir::failure();
    }

    // IDUConvContinue
    if (buildIDUConvContinue(_rewriter, _origVarOp.getLoc(), _log, origInvOp.getIsContinued()).failed()) {
        return mlir::failure();
    }

    // IDUBinaryConfig
    // variant.offset_addr.offset_addr_bf.bin_cfg not set in nce_lib
    // leaving field value to reset initialization in lowering VPUASM2NPUReg40XX

    return mlir::success();
}

}  // namespace VPUIPDPU
}  // namespace vpux
