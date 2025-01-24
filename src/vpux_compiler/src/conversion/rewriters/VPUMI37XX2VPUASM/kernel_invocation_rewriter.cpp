//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/kernel_invocation_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi37xx2vpuasm {

mlir::FailureOr<SymbolizationResult> KernelInvocationRewriter::symbolize(
        VPUMI37XX::ActKernelInvocationOp op, SymbolMapper&, mlir::ConversionPatternRewriter& rewriter) const {
    auto symName = findSym(op).getRootReference();
    auto taskLocation = findSym(op.getTaskLocation());

    auto kernelRange = findSym(op.getRangeIndex());

    // E#69737:: change VPUMI37XX to have op-operand relationship between Invo's and Param's
    auto parentFunc = op.getOperation()->getParentOfType<mlir::func::FuncOp>();
    VPUX_THROW_WHEN(parentFunc == nullptr, "Invocation op not in a FuncOp");
    auto kernelParamsOps = parentFunc.getOps<vpux::VPUMI37XX::KernelParamsOp>();
    auto params =
            std::find_if(kernelParamsOps.begin(), kernelParamsOps.end(), [&op](VPUMI37XX::KernelParamsOp paramsOp) {
                return paramsOp.getType().getValue() == op.getType().getValue();
            });
    auto kernelParamsSym = findSym((*params).getResult());

    auto waitAttr = vectorizeBarriers(op.getWaitBarriers());
    auto updateAttr = vectorizeBarriers(op.getUpdateBarriers());

    auto taskIdx = mlir::TypeAttr::get(op.getType());

    auto oldKernelRange = op.getRangeIndex().getDefiningOp<VPUMI37XX::ActKernelRangeOp>();
    auto kernelIndexAttr =
            mlir::IntegerAttr::get(vpux::getUInt64Type(getContext()), oldKernelRange.getType().getValue());

    auto kernelData = findSym(oldKernelRange.getKernelArgsIndex());

    mlir::SymbolRefAttr profilingData;
    if (op.getProfilingData()) {
        profilingData = findSym(op.getProfilingData());
    }

    auto newOp = rewriter.create<VPUASM::ActKernelInvocationOp>(
            op.getLoc(), symName, taskIdx, taskLocation, /* next_link */ nullptr, kernelRange, kernelData,
            kernelParamsSym, waitAttr, updateAttr, profilingData, op.getTileAttr(), op.getStartAfterAttr(),
            op.getCleanAfterAttr(), kernelIndexAttr);

    rewriter.eraseOp(op);

    return SymbolizationResult(newOp);
}

}  // namespace vpumi37xx2vpuasm
}  // namespace vpux
