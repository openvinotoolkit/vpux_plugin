//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/m2i_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> M2IRewriter::symbolize(VPUMI40XX::M2IOp op, SymbolMapper& mapper,
                                                            mlir::ConversionPatternRewriter& rewriter) const {
    auto result = op.getResult();
    mlir::StringAttr symName = findSym(result).getRootReference();

    auto optionalSym = [&](mlir::Value val) -> mlir::SymbolRefAttr {
        return val ? findSym(val) : nullptr;
    };

    auto taskLocation = optionalSym(op.getTaskLocation());
    auto taskIdx = mlir::TypeAttr::get(op.getType());

    auto nextM2IIt = std::find_if(result.user_begin(), result.user_end(), [](mlir::Operation* op) -> bool {
        return mlir::isa<VPUMI40XX::M2IOp>(op);
    });

    mlir::SymbolRefAttr nextLink = nullptr;
    if (nextM2IIt != result.user_end()) {
        auto nextM2I = mlir::cast<VPUMI40XX::M2IOp>(*nextM2IIt);
        auto nextLinkIt = mapper.find(nextM2I.getTaskLocation());
        VPUX_THROW_WHEN(nextLinkIt == mapper.end(), "Cannot find symbol name entry for {0}",
                        nextM2I.getOperationName());
        nextLink = nextLinkIt->getSecond();
    }

    auto inputSym = findSym(op.getInput());
    auto outputSym = findSym(op.getOutputBuff());
    auto profilingDataSym = op.getProfilingData() != nullptr ? findSym(op.getProfilingData()) : nullptr;

    auto waitAttr = vectorizeBarriers(op.getWaitBarriers());
    auto updateAttr = vectorizeBarriers(op.getUpdateBarriers());

    auto startAfter = op.getStartAfterAttr();
    auto cleanAfter = op.getCleanAfterAttr();

    auto newOp = rewriter.create<VPUASM::M2IOp>(
            op.getLoc(), symName, taskIdx, taskLocation, nextLink, inputSym, outputSym, profilingDataSym,
            op.getDoCscAttr(), op.getDoNormAttr(), op.getInFmtAttr(), op.getOutFmtAttr(),
            op.getChromaInReverseChannelsAttr(), op.getChromaOutReverseChannelsAttr(),
            op.getLumaInReverseChannelsAttr(), op.getLumaOutReverseChannelsAttr(), op.getScaleFactorXAttr(),
            op.getScaleFactorYAttr(), op.getNormAttr(), op.getTileOffsetXAttr(), op.getTileOffsetYAttr(),
            op.getInterpAttr(), waitAttr, updateAttr, startAfter, cleanAfter);

    rewriter.eraseOp(op);

    return SymbolizationResult(newOp);
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
