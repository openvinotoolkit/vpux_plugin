//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/view_task_range_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> ViewTaskRangeRewriter::symbolize(VPURegMapped::ViewTaskRangeOp op,
                                                                      SymbolMapper& mapper,
                                                                      mlir::ConversionPatternRewriter& rewriter) const {
    mapper[op.getResult()] = mapper[op.getFirst()];

    rewriter.eraseOp(op);
    return SymbolizationResult();
}

llvm::SmallVector<mlir::FlatSymbolRefAttr> ViewTaskRangeRewriter::getSymbolicNames(VPURegMapped::ViewTaskRangeOp op,
                                                                                   size_t) {
    return {mlir::dyn_cast<mlir::FlatSymbolRefAttr>(findSym(op.getFirst()))};
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
