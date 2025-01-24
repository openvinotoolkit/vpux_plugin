//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/enqueue_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> EnqueueRewriter::symbolize(VPURegMapped::EnqueueOp op, SymbolMapper&,
                                                                mlir::ConversionPatternRewriter& rewriter) const {
    auto result = op.getResult();
    auto symName = findSym(result).getRootReference();
    auto taskIdx = mlir::TypeAttr::get(op.getType());

    if (op.getStart() != op.getEnd()) {
        return op.emitOpError("Rewriting enqueueOp with the range > 1");
    }

    // we enqueue the location projection of the first task
    auto firstTask = mlir::cast<VPURegMapped::TaskOpInterface>(op.getStart().getDefiningOp());
    auto firstTaskSym =
            firstTask.getTaskLocation() ? findSym(firstTask.getTaskLocation()) : findSym(firstTask.getResult());
    auto count = op.getEnd().getType().cast<VPURegMapped::IndexType>().getValue() -
                 op.getStart().getType().cast<VPURegMapped::IndexType>().getValue() + 1;

    auto realTaskIdx = mlir::TypeAttr::get(op.getStart().getType());

    auto newOp =
            rewriter.create<VPUASM::WorkItemOp>(op.getLoc(), symName, taskIdx, realTaskIdx, op.getTaskTypeAttr(),
                                                firstTaskSym, rewriter.getI64IntegerAttr(static_cast<int64_t>(count)));
    rewriter.eraseOp(op);

    return SymbolizationResult(newOp);
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
