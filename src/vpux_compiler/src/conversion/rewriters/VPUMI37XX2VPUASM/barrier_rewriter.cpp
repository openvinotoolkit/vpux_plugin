//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/barrier_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi37xx2vpuasm {

mlir::FailureOr<SymbolizationResult> BarrierRewriter::symbolize(VPUMI37XX::ConfigureBarrierOp op, SymbolMapper&,
                                                                mlir::ConversionPatternRewriter& rewriter) const {
    auto result = op.getResult();
    auto symName = findSym(result).getRootReference();
    auto taskIdx = mlir::TypeAttr::get(op.getType());

    // VPUASM::ConfigureBarrierOp has an optional attribute work_item_idx which is not using
    // in 37XX generation. For avoid code duplication across operation lets set it as nullptr here
    auto newOp = rewriter.create<VPUASM::ConfigureBarrierOp>(op.getLoc(), symName, taskIdx, nullptr, op.getIdAttr(),
                                                             op.getNextSameIdAttr(), op.getProducerCountAttr(),
                                                             op.getConsumerCountAttr());
    rewriter.eraseOp(op);

    return SymbolizationResult(newOp);
}

}  // namespace vpumi37xx2vpuasm
}  // namespace vpux
