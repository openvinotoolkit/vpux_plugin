//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/task_sink_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::LogicalResult TaskSinkRewriter::symbolize(VPURegMapped::TaskSinkOp op, SymbolMapper&,
                                                mlir::ConversionPatternRewriter& rewriter) const {
    auto symName = findSym(op.getResult());

    rewriter.create<VPUASM::TaskSinkOp>(op.getLoc(), symName.getAttr(), op.getTileAttr(), op.getTaskTypeAttr());
    rewriter.eraseOp(op);
    return mlir::success();
}

mlir::FlatSymbolRefAttr TaskSinkRewriter::getSymbolicName(VPURegMapped::TaskSinkOp op, size_t counter) {
    auto fullName = VPURegMapped::TaskSinkOp::getOperationName();
    auto opName = fullName.drop_front(VPURegMapped::VPURegMappedDialect::getDialectNamespace().size() + 1);

    auto index = std::to_string(counter);
    auto symName = mlir::StringAttr::get(op.getContext(), opName + index);
    return mlir::FlatSymbolRefAttr::get(symName);
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
