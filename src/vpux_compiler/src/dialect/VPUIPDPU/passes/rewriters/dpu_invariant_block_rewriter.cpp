//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/dpu_invariant_block_rewriters.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"

namespace vpux {
namespace VPUIPDPU {

DPUInvariantBlockRewriter::DPUInvariantBlockRewriter(VPUASM::DPUInvariantOp origInvOp, mlir::Block* invBlock,
                                                     std::map<BlockArg, size_t>& invBlockArgsPos,
                                                     mlir::PatternRewriter& rewriter, const Logger& log)
        : _origInvOp(origInvOp),
          _invBlock(invBlock),
          _invBlockArgsPos(invBlockArgsPos),
          _rewriter(rewriter),
          _log(log) {
}

mlir::LogicalResult DPUInvariantBlockRewriter::insertInvBlockArgs(VPUASM::DPUInvariantOp op, mlir::Block* invBlock,
                                                                  std::map<BlockArg, size_t>& invBlockArgsPos,
                                                                  const Logger& log,
                                                                  ELF::SymbolReferenceMap& symRefMap) {
    // input activations
    auto inputType = getBufferType(symRefMap.lookupSymbol(op.getInput()));
    invBlock->addArgument(inputType, op.getLoc());
    invBlockArgsPos[BlockArg::ACT_IN] = invBlock->getNumArguments() - 1;

    // input storage elements
    if (op.getInputStorageElementTable()) {
        auto inputSETableType = getBufferType(symRefMap.lookupSymbol(op.getInputStorageElementTable().value()));
        invBlock->addArgument(inputSETableType, op.getLoc());
        invBlockArgsPos[BlockArg::ACT_SE_IN] = invBlock->getNumArguments() - 1;
    }

    // input sparsity
    if (op.getInputSparsityMap()) {
        auto inputSparsityMapType = getBufferType(symRefMap.lookupSymbol(op.getInputSparsityMap().value()));
        invBlock->addArgument(inputSparsityMapType, op.getLoc());
        invBlockArgsPos[BlockArg::ACT_SPARSE_MAP_IN] = invBlock->getNumArguments() - 1;
    }

    // weights table
    if (op.getWeightTable()) {
        auto weightTableType = getBufferType(symRefMap.lookupSymbol(op.getWeightTable().value()));
        invBlock->addArgument(weightTableType, op.getLoc());
        invBlockArgsPos[BlockArg::WEIGHTS_TABLE] = invBlock->getNumArguments() - 1;
    }

    // weights
    if (op.getWeights()) {
        auto weightsType = getBufferType(symRefMap.lookupSymbol(op.getWeights().value()));
        invBlock->addArgument(weightsType, op.getLoc());
        invBlockArgsPos[BlockArg::WEIGHTS] = invBlock->getNumArguments() - 1;
    }

    // weights sparsity
    if (op.getWeightsSparsityMap()) {
        auto weightsSparsityMapType = getBufferType(symRefMap.lookupSymbol(op.getWeightsSparsityMap().value()));
        invBlock->addArgument(weightsSparsityMapType, op.getLoc());
        invBlockArgsPos[BlockArg::WEIGHTS_SPARSE_MAP] = invBlock->getNumArguments() - 1;
    }

    // spr lookup table
    if (op.getSprLookupTable()) {
        auto sprLookupTableType = getBufferType(symRefMap.lookupSymbol(op.getSprLookupTable().value()));
        invBlock->addArgument(sprLookupTableType, op.getLoc());
        invBlockArgsPos[BlockArg::SPR_LOOKUP_TABLE] = invBlock->getNumArguments() - 1;
    }

    // output activations
    mlir::MemRefType outType;
    if (!op.getIsContinued() && op.getOutput()) {
        outType = getBufferType(symRefMap.lookupSymbol(op.getOutput().value()));
    } else if (op.getIsContinued() && op.getOutputTypeContinued()) {
        outType = op.getOutputTypeContinued().value().getMemref();
    } else {
        log.error("Expected either output buffer or output type for continued mode");
        return mlir::failure();
    }
    invBlock->addArgument(outType, op.getLoc());
    invBlockArgsPos[BlockArg::ACT_OUT] = invBlock->getNumArguments() - 1;

    // output sparsity
    if (op.getOutputSparsityMap()) {
        auto outputSparsityMapType = getBufferType(symRefMap.lookupSymbol(op.getOutputSparsityMap().value()));
        invBlock->addArgument(outputSparsityMapType, op.getLoc());
        invBlockArgsPos[BlockArg::ACT_SPARSE_MAP_OUT] = invBlock->getNumArguments() - 1;
    }

    return mlir::success();
}

mlir::Type DPUInvariantBlockRewriter::getBaseType(mlir::Type type) {
    if (type.isa<mlir::quant::QuantizedType>()) {
        auto quantType = type.cast<mlir::quant::QuantizedType>();
        auto quantStorageType = quantType.getStorageType();
        if (quantStorageType.isFloat8E5M2()) {
            return mlir::Float8E5M2Type::get(type.getContext());
        }

        if (quantStorageType.isFloat8E4M3FN()) {
            return mlir::Float8E4M3FNType::get(type.getContext());
        }

        auto signedness = quantType.isSigned() ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned;
        auto bitWidth = quantType.getStorageTypeIntegralWidth();
        return mlir::IntegerType::get(type.getContext(), bitWidth, signedness);
    }

    return type;
}

mlir::LogicalResult DPUInvariantBlockRewriter::getQuantConfig(const Logger&, mlir::Type type,
                                                              SmallVector<int64_t>& quantMult,
                                                              SmallVector<int64_t>& quantShift,
                                                              SmallVector<uint8_t>& quantZero) {
    if (const auto qType = type.dyn_cast<mlir::quant::UniformQuantizedType>()) {
        quantZero.push_back(checked_cast<uint8_t>(qType.getZeroPoint()));
        const auto scaleApproximation = QuantizationApproximation(VPU::ArchKind::NPU40XX, qType.getScale());
        quantMult.push_back(scaleApproximation.mult());
        quantShift.push_back(scaleApproximation.shift());
    } else if (const auto qPerAxisType = type.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
        auto qtypeQuantZp = qPerAxisType.getZeroPoints();
        auto qtypeQuantScale = qPerAxisType.getScales();

        quantZero.resize(qtypeQuantZp.size());
        std::transform(qtypeQuantZp.begin(), qtypeQuantZp.end(), quantZero.begin(), [](int64_t val) {
            return checked_cast<uint8_t>(val);
        });

        quantMult.resize(qtypeQuantScale.size());
        quantShift.resize(qtypeQuantScale.size());
        for (std::size_t i = 0; i < qtypeQuantScale.size(); ++i) {
            const auto scaleApproximation = QuantizationApproximation(VPU::ArchKind::NPU40XX, qtypeQuantScale[i]);
            quantMult[i] = scaleApproximation.mult();
            quantShift[i] = scaleApproximation.shift();
        }
    } else {
        quantMult.push_back(1);
        quantShift.push_back(0);
        quantZero.push_back(0);
    }

    return mlir::success();
}

mlir::BlockArgument DPUInvariantBlockRewriter::getInvBlockArg(BlockArg invBlockArg) const {
    auto arg = _invBlockArgsPos.find(invBlockArg);
    if (arg == _invBlockArgsPos.end()) {
        return nullptr;
    }

    return _invBlock->getArgument(arg->second);
}

}  // namespace VPUIPDPU
}  // namespace vpux
