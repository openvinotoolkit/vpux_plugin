//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/dpu_invariant_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi37xx2vpuasm {

mlir::FailureOr<SymbolizationResult> DPUInvariantRewriter::symbolize(VPUMI37XX::DPUInvariantOp op, SymbolMapper&,
                                                                     mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = getContext();
    auto symName = findSym(op).getRootReference();
    auto taskLocation = findSym(op.getTaskLocation());

    auto optionalSym = [&](mlir::Value val) -> mlir::SymbolRefAttr {
        auto sym = val ? findSym(val) : nullptr;
        return sym;
    };

    auto inputSym = findSym(op.getInput());
    auto inputSparsityMapSym = optionalSym(op.getInputSparsityMap());
    auto inputSETableSym = optionalSym(op.getInputStorageElementTable());

    auto weightsSym = optionalSym(op.getWeights());
    auto weightsSparsityMapSym = optionalSym(op.getWeightsSparsityMap());
    auto weightTableSym = optionalSym(op.getWeightTable());

    auto parentInputSym = findSym(op.getParentInput());
    auto parentInputSparsityMapSym = optionalSym(op.getParentInputSparsityMap());
    auto parentInputSETableSym = optionalSym(op.getParentInputSparsityMap());

    auto parentOutputSym = findSym(op.getParentOutput());

    llvm::SmallVector<mlir::Attribute, 6> outputSyms(op.getOutputBuffs().size());
    for (auto outputIt : llvm::enumerate(op.getOutputBuffs())) {
        auto outputSym = findSym(outputIt.value());
        outputSyms[outputIt.index()] = outputSym;
    }
    auto outputSymsAttr = mlir::ArrayAttr::get(ctx, outputSyms);

    auto profilingDataSym = optionalSym(op.getProfilingData());

    auto waitAttr = vectorizeBarriers(op.getWaitBarriers());
    auto updateAttr = vectorizeBarriers(op.getUpdateBarriers());

    auto taskIdx = mlir::TypeAttr::get(op.getType());

    auto invariant = rewriter.create<VPUASM::DPUInvariantOp_37XX>(
            op.getLoc(), symName, taskIdx, taskLocation, inputSym, inputSparsityMapSym, inputSETableSym, weightsSym,
            weightsSparsityMapSym, weightTableSym, parentInputSym, parentInputSparsityMapSym, parentInputSETableSym,
            parentOutputSym, outputSymsAttr, profilingDataSym, waitAttr, updateAttr, op.getNceTaskTypeAttr(),
            op.getEltwiseTypeAttr(), op.getMpeFrequentModeAttr(), op.getKernelSizeAttr(), op.getKernelStridesAttr(),
            op.getKernelPaddingAttr(), op.getIsContinuedAttr(), op.getCmSpPatternAttr(),
            op.getInputChannelsCompressionAttr(), op.getIsZeroOffsetWeightsTableAttr(), op.getIsSegmentedAttr(),
            op.getOutChannelOffsetAttr(), op.getStartAfterAttr(), op.getCleanAfterAttr());

    {
        auto& ppeRegion = invariant.getPpe();
        ppeRegion.emplaceBlock();

        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToEnd(&ppeRegion.front());

        for (auto ppe : op.getPpe().getOps<VPUMI37XX::PPETaskOp>()) {
            rewriter.create<VPUASM::PPETaskOp>(rewriter.getUnknownLoc(), ppe->getResultTypes(), ppe->getOperands(),
                                               ppe->getAttrDictionary().getValue());
        }
    }
    rewriter.eraseOp(op);

    return SymbolizationResult(invariant);
}

}  // namespace vpumi37xx2vpuasm
}  // namespace vpux
