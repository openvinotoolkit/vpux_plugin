//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/kernel_invocation_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> KernelInvocationRewriter::symbolize(
        VPUMI40XX::ActKernelInvocationOp op, SymbolMapper&, mlir::ConversionPatternRewriter& rewriter) const {
    auto symName = findSym(op).getRootReference();
    auto taskLocation = findSym(op.getTaskLocation());
    auto kernelParams = findSym(op.getKernelParams());

    auto waitAttr = vectorizeBarriers(op.getWaitBarriers());
    auto updateAttr = vectorizeBarriers(op.getUpdateBarriers());

    auto taskIdx = mlir::TypeAttr::get(op.getType());

    auto oldKernelRange = op.getRangeIndex().getDefiningOp<VPUMI40XX::ActKernelRangeOp>();
    auto kernelIndexAttr =
            mlir::IntegerAttr::get(vpux::getUInt64Type(getContext()), oldKernelRange.getIndexType().getValue());

    auto kernelTaskType = oldKernelRange.getKernelTaskType();
    bool isCacheOp = VPUIP::isCacheOpTaskType(kernelTaskType, /*includePrefetch=*/false);

    auto kernelData = isCacheOp ? nullptr : findSym(oldKernelRange.getKernelArgsIndex());
    auto kernelRange = findSym(oldKernelRange.getTaskLocation());

    mlir::SymbolRefAttr profilingData = nullptr;
    if (auto profBuffer = op.getProfilingData()) {
        profilingData = findSym(profBuffer);
    }

    mlir::SymbolRefAttr nextLink = nullptr;
    auto nextInvocation = VPUMI40XX::getNextOp(op);
    while (nextInvocation) {
        if (auto nextInvocationTaskLink = nextInvocation.getTaskLink();
            nextInvocationTaskLink.has_value() && nextInvocationTaskLink.value() == op.getType()) {
            nextLink = findSym(nextInvocation.getTaskLocation());
            break;
        }
        nextInvocation = VPUMI40XX::getNextOp(nextInvocation);
    }

    auto newOp = rewriter.create<VPUASM::ActKernelInvocationOp>(
            op.getLoc(), symName, taskIdx, taskLocation, nextLink, kernelRange, kernelData, kernelParams, waitAttr,
            updateAttr, profilingData, op.getTileAttr(), op.getStartAfterAttr(), op.getCleanAfterAttr(),
            kernelIndexAttr);

    rewriter.eraseOp(op);

    return SymbolizationResult(newOp);
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
