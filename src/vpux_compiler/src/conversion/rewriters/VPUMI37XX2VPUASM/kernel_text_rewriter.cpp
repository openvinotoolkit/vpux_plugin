//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/kernel_text_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi37xx2vpuasm {

mlir::FailureOr<SymbolizationResult> KernelTextRewriter::symbolize(VPUMI37XX::DeclareKernelTextOp op, SymbolMapper&,
                                                                   mlir::ConversionPatternRewriter& rewriter) const {
    auto symName = findSym(op).getRootReference();
    auto newOp = rewriter.create<VPUASM::DeclareKernelTextOp>(op.getLoc(), symName, op.getKernelPathAttr());
    rewriter.eraseOp(op);
    return SymbolizationResult(newOp);
}

}  // namespace vpumi37xx2vpuasm
}  // namespace vpux
