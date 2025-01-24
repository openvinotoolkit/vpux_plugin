//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/declare_buffer_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi37xx2vpuasm {

mlir::FailureOr<SymbolizationResult> DeclareBufferRewriter::symbolize(VPURT::DeclareBufferOp op, SymbolMapper& mapper,
                                                                      mlir::ConversionPatternRewriter& rewriter) const {
    mlir::MLIRContext* ctx = rewriter.getContext();
    auto result = op.getResult();
    auto symNameIt = mapper.find(result);
    VPUX_THROW_WHEN(symNameIt == mapper.end(), "Could not find symbol name entry for {0}", op.getOperationName());

    auto symName = symNameIt->getSecond().getRootReference();

    if (result.getType().isa<mlir::MemRefType>()) {
        auto bufferSec = op.getSection();
        auto sectionIndex = op.getSectionIndex();
        uint64_t bufferIdx = sectionIndex.has_value() ? sectionIndex.value()[0].cast<mlir::IntegerAttr>().getInt() : 0;
        auto bufferOffs = op.getByteOffset();

        auto memLocation = VPUASM::MemLocationType::get(ctx, bufferSec, bufferIdx, bufferOffs);
        auto memref = result.getType().cast<mlir::MemRefType>();
        auto traits = VPUASM::BufferTraitsType::get(ctx, op.getSwizzlingKey().value_or(0));

        auto buffType = VPUASM::BufferType::get(ctx, memLocation, memref, traits);

        auto newOp = rewriter.create<VPUASM::DeclareBufferOp>(op.getLoc(), symName, buffType);

        rewriter.eraseOp(op);

        return SymbolizationResult(newOp);
    } else {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.startOpModification(op);
        rewriter.setInsertionPointAfter(op);
        rewriter.create<VPUASM::SymbolizeValueOp>(op.getLoc(), result, symName);
        rewriter.finalizeOpModification(op);
        return SymbolizationResult();
    }
}

llvm::SmallVector<mlir::FlatSymbolRefAttr> DeclareBufferRewriter::getSymbolicNames(VPURT::DeclareBufferOp op,
                                                                                   size_t counter) {
    auto fullName = VPURT::DeclareBufferOp::getOperationName();
    auto opName = fullName.drop_front(VPURT::VPURTDialect::getDialectNamespace().size() + 1);

    auto index = std::to_string(counter);
    auto symName = mlir::StringAttr::get(op.getContext(), opName + index);
    return {mlir::FlatSymbolRefAttr::get(symName)};
}

}  // namespace vpumi37xx2vpuasm
}  // namespace vpux
