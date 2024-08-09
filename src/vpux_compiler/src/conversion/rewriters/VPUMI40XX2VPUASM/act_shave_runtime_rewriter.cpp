//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/act_shave_runtime_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

llvm::SmallVector<mlir::FlatSymbolRefAttr> ActShaveRtRewriter::getSymbolicNames(VPUMI40XX::ActShaveRtOp, size_t) {
    return {mlir::FlatSymbolRefAttr::get(getContext(), "ActShaveRt")};
}

mlir::LogicalResult ActShaveRtRewriter::symbolize(VPUMI40XX::ActShaveRtOp op, SymbolMapper&,
                                                  mlir::ConversionPatternRewriter& rewriter) const {
    auto result = op.getResult();
    auto symName = findSym(result).getRootReference();

    rewriter.create<VPUASM::ActShaveRtOp>(op.getLoc(), symName, op.getKernelPathAttr());

    rewriter.eraseOp(op);

    return mlir::success();
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
