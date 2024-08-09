//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/mapped_inference_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi37xx2vpuasm {

llvm::SmallVector<mlir::FlatSymbolRefAttr> MappedInferenceRewriter::getSymbolicNames(VPUMI37XX::MappedInferenceOp,
                                                                                     size_t) {
    return {mlir::FlatSymbolRefAttr::get(getContext(), "MappedInference")};
}

mlir::LogicalResult MappedInferenceRewriter::symbolize(VPUMI37XX::MappedInferenceOp op, SymbolMapper&,
                                                       mlir::ConversionPatternRewriter& rewriter) const {
    mlir::MLIRContext* ctx = rewriter.getContext();
    auto result = op.getResult();

    llvm::SmallVector<mlir::Attribute> dmas(op.getDmaTasks().size());
    for (auto dma : llvm::enumerate(op.getDmaTasks())) {
        auto dmaName = findSym(dma.value());

        dmas[dma.index()] = dmaName;
    }

    llvm::SmallVector<mlir::Attribute> shaveStacks(op.getActShaveStacks().size());
    for (auto shaveStack : llvm::enumerate(op.getActShaveStacks())) {
        auto shaveStackName = findSym(shaveStack.value());

        shaveStacks[shaveStack.index()] = shaveStackName;
    }

    mlir::StringAttr symName = findSym(result).getRootReference();
    mlir::ArrayAttr dmasAttr = dmas.size() ? mlir::ArrayAttr::get(ctx, dmas) : nullptr;
    mlir::SymbolRefAttr invariantTasks = op.getInvariantTasks() ? findSym(op.getInvariantTasks()) : nullptr;
    mlir::SymbolRefAttr variantTasks = op.getVariantTasks() ? findSym(op.getVariantTasks()) : nullptr;
    mlir::SymbolRefAttr actKernelRanges = op.getActKernelRanges() ? findSym(op.getActKernelRanges()) : nullptr;
    mlir::SymbolRefAttr actKernelInvocations =
            op.getActKernelInvocations() ? findSym(op.getActKernelInvocations()) : nullptr;
    mlir::SymbolRefAttr barrierTasks = op.getBarrierTasks() ? findSym(op.getBarrierTasks()) : nullptr;
    mlir::SymbolRefAttr actShaveRT = op.getActShaveRt() ? findSym(op.getActShaveRt()) : nullptr;
    mlir::ArrayAttr actShaveStacks = shaveStacks.size() ? mlir::ArrayAttr::get(ctx, shaveStacks) : nullptr;

    rewriter.create<VPUASM::MappedInferenceOp_37XX>(op.getLoc(), symName, dmasAttr, invariantTasks, variantTasks,
                                                    actKernelRanges, actKernelInvocations, barrierTasks, actShaveRT,
                                                    actShaveStacks, op.getDmaCountAttr(), op.getInvariantCountAttr(),
                                                    op.getVariantCountAttr(), op.getActKernelRangesCountAttr(),
                                                    op.getActKernelInvocationsCountAttr(), op.getBarrierCountAttr());

    rewriter.eraseOp(op);

    return mlir::success();
}

}  // namespace vpumi37xx2vpuasm
}  // namespace vpux
