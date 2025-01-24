//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/declare_task_addr_buffer_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> DeclareTaskAddrBuffRewriter::symbolize(
        VPURegMapped::DeclareTaskAddrBufferOp op, SymbolMapper& mapper,
        mlir::ConversionPatternRewriter& rewriter) const {
    mlir::MLIRContext* ctx = rewriter.getContext();
    auto result = op.getResult();
    auto symNameIt = mapper.find(result);
    VPUX_THROW_WHEN(symNameIt == mapper.end(), "Could not find symbol name entry for {0}", op.getOperationName());

    auto symName = symNameIt->getSecond().getRootReference();

    auto first = findSym(op.getFirst());
    auto last = findSym(op.getLast());

    auto ndType = op.getType().cast<vpux::NDTypeInterface>();
    auto memSpace = VPURT::symbolizeBufferSection(ndType.getMemSpace().getLeafName());
    auto memIdx = ndType.getMemSpace().getIndex().value_or(0);
    auto memLocation = VPUASM::MemLocationType::get(ctx, memSpace.value_or(VPURT::BufferSection::DDR), memIdx,
                                                    op.getOffset().value_or(0));

    auto memref = op.getType();
    auto traits = VPUASM::BufferTraitsType::get(ctx, 0);

    auto bufferType = VPUASM::BufferType::get(ctx, memLocation, memref, traits);

    // TODO: E101726
    auto taskType = VPURegMapped::TaskType::DPUVariant;
    auto newOp = rewriter.create<VPUASM::DeclareTaskAddrBufOp>(op.getLoc(), symName, first, last, bufferType, taskType);
    rewriter.eraseOp(op);

    return SymbolizationResult(newOp);
}

llvm::SmallVector<mlir::FlatSymbolRefAttr> DeclareTaskAddrBuffRewriter::getSymbolicNames(
        VPURegMapped::DeclareTaskAddrBufferOp op, size_t counter) {
    auto fullName = VPURegMapped::DeclareTaskAddrBufferOp::getOperationName();
    auto opName = fullName.drop_front(VPURegMapped::VPURegMappedDialect::getDialectNamespace().size() + 1);

    auto index = std::to_string(counter);
    auto symName = mlir::StringAttr::get(op.getContext(), opName + index);
    return {mlir::FlatSymbolRefAttr::get(symName)};
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
