//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/declare_task_buffer_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::LogicalResult DeclareTaskBufferRewriter::symbolize(VPURegMapped::DeclareTaskBufferOp op, SymbolMapper&,
                                                         mlir::ConversionPatternRewriter& rewriter) const {
    auto symName = findSym(op.getResult()).getRootReference();
    auto taskIdx = mlir::TypeAttr::get(op.getType());

    rewriter.create<VPUASM::DeclareTaskBufferOp>(op.getLoc(), symName, taskIdx, op.getTaskTypeAttr(),
                                                 op.getOffsetAttr());
    rewriter.eraseOp(op);
    return mlir::success();
}

llvm::SmallVector<mlir::FlatSymbolRefAttr> DeclareTaskBufferRewriter::getSymbolicNames(
        VPURegMapped::DeclareTaskBufferOp op, size_t) {
    auto opName = op->getName().stripDialect();
    auto taskTypeString = VPURegMapped::stringifyTaskType(op.getTaskType());

    auto tileIdx = std::to_string(op.getType().getTileIdx());
    auto srcTypeIdx = std::to_string(op.getType().getListIdx());
    auto opIdx = std::to_string(op.getType().getValue());

    auto symName = mlir::StringAttr::get(
            op.getContext(), opName + "_" + taskTypeString + "_" + tileIdx + "_" + srcTypeIdx + "_" + opIdx);

    return {mlir::FlatSymbolRefAttr::get(symName)};
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
