//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/kernel_range_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi37xx2vpuasm {

mlir::LogicalResult KernelRangeRewriter::symbolize(VPUMI37XX::ActKernelRangeOp op, SymbolMapper&,
                                                   mlir::ConversionPatternRewriter& rewriter) const {
    auto symName = findSym(op).getRootReference();
    auto taskLocation = findSym(op.getTaskLocation());

    auto kernelTextAttr = findSym(op.getKernelTextIndex());
    auto kernelEntryAttr = findSym(op.getKernelEntryIndex());

    auto kernelTaskType = op.getKernelTaskType();
    auto taskIdx = mlir::TypeAttr::get(op.getType());

    rewriter.create<VPUASM::ActKernelRangeOp>(op.getLoc(), symName, taskIdx, taskLocation, kernelTextAttr,
                                              kernelEntryAttr, kernelTaskType);
    rewriter.eraseOp(op);

    return mlir::success();
}

}  // namespace vpumi37xx2vpuasm
}  // namespace vpux
