//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/bootstrap_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::LogicalResult BootstrapRewriter::symbolize(VPUMI40XX::BootstrapOp op, SymbolMapper&,
                                                 mlir::ConversionPatternRewriter& rewriter) const {
    auto result = op.getResult();
    auto symName = findSym(result).getRootReference();
    int barrierId = op.getBarrier().getType().cast<VPURegMapped::IndexType>().getValue();
    rewriter.create<VPUASM::BootstrapOp>(op.getLoc(), symName, barrierId);
    rewriter.eraseOp(op);

    return mlir::success();
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
