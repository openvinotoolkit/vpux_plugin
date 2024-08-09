//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/view_task_range_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::LogicalResult ViewTaskRangeRewriter::symbolize(VPURegMapped::ViewTaskRangeOp op, SymbolMapper&,
                                                     mlir::ConversionPatternRewriter& rewriter) const {
    rewriter.eraseOp(op);
    return mlir::success();
}

llvm::SmallVector<mlir::FlatSymbolRefAttr> ViewTaskRangeRewriter::getSymbolicNames(VPURegMapped::ViewTaskRangeOp op,
                                                                                   size_t) {
    return {findSym(op.getFirst())};
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
