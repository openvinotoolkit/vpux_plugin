//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/mapped_inference_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FlatSymbolRefAttr MappedInferenceRewriter::getSymbolicName(VPUMI40XX::MappedInferenceOp, size_t) {
    return mlir::FlatSymbolRefAttr::get(getContext(), "MappedInference");
}

mlir::LogicalResult MappedInferenceRewriter::symbolize(VPUMI40XX::MappedInferenceOp op, SymbolMapper&,
                                                       mlir::ConversionPatternRewriter& rewriter) const {
    mlir::MLIRContext* ctx = rewriter.getContext();
    auto result = op.getResult();

    llvm::SmallVector<llvm::SmallVector<mlir::Attribute>> dmas(op.getDmaTasks().size());
    for (auto tileDmas : llvm::enumerate(op.getDmaTasks())) {
        dmas[tileDmas.index()].resize(tileDmas.value().size());
        for (auto dma : llvm::enumerate(tileDmas.value())) {
            auto dmaName = findSym(dma.value());

            dmas[tileDmas.index()][dma.index()] = dmaName;
        }
    }

    llvm::SmallVector<mlir::Attribute> invariantTasks(op.getInvariantTasks().size());
    for (auto invariantTask : llvm::enumerate(op.getInvariantTasks())) {
        auto invariantTaskName = findSym(invariantTask.value());

        invariantTasks[invariantTask.index()] = invariantTaskName;
    }

    llvm::SmallVector<mlir::Attribute> variantTasks(op.getVariantTasks().size());
    for (auto variantTask : llvm::enumerate(op.getVariantTasks())) {
        auto variantTaskName = findSym(variantTask.value());

        variantTasks[variantTask.index()] = variantTaskName;
    }

    llvm::SmallVector<mlir::Attribute> actKernelRanges(op.getActKernelRanges().size());
    for (auto actKernelRange : llvm::enumerate(op.getActKernelRanges())) {
        auto actKernelRangeName = findSym(actKernelRange.value());

        actKernelRanges[actKernelRange.index()] = actKernelRangeName;
    }

    llvm::SmallVector<mlir::Attribute> actKernelInvocations(op.getActKernelInvocations().size());
    for (auto actKernelInvocation : llvm::enumerate(op.getActKernelInvocations())) {
        auto actKernelInvocationName = findSym(actKernelInvocation.value());

        actKernelInvocations[actKernelInvocation.index()] = actKernelInvocationName;
    }

    llvm::SmallVector<mlir::Attribute> actShaveStacks(op.getActShaveStacks().size());
    for (auto actShaveStack : llvm::enumerate(op.getActShaveStacks())) {
        auto actShaveStacksName = findSym(actShaveStack.value());

        actShaveStacks[actShaveStack.index()] = actShaveStacksName;
    }

    SmallVector<mlir::Attribute> dmasAttrVec;
    for (const auto& dma : dmas) {
        dmasAttrVec.push_back(mlir::ArrayAttr::get(ctx, dma));
    }

    mlir::StringAttr symName = findSym(result).getRootReference();
    mlir::ArrayAttr dmasAttr = dmasAttrVec.size() ? mlir::ArrayAttr::get(ctx, dmasAttrVec) : nullptr;
    mlir::ArrayAttr invariantTasksAttr = invariantTasks.size() ? mlir::ArrayAttr::get(ctx, invariantTasks) : nullptr;
    mlir::ArrayAttr variantTasksAttr = variantTasks.size() ? mlir::ArrayAttr::get(ctx, variantTasks) : nullptr;
    mlir::ArrayAttr actKernelRangesAttr = actKernelRanges.size() ? mlir::ArrayAttr::get(ctx, actKernelRanges) : nullptr;
    mlir::ArrayAttr actKernelInvocationsAttr =
            actKernelInvocations.size() ? mlir::ArrayAttr::get(ctx, actKernelInvocations) : nullptr;
    mlir::SymbolRefAttr mediaTasksAttr = op.getMediaTasks() ? findSym(op.getMediaTasks()) : nullptr;
    mlir::SymbolRefAttr barrierTasksAttr = op.getBarrierTasks() ? findSym(op.getBarrierTasks()) : nullptr;
    mlir::SymbolRefAttr actShaveRtAttr = op.getActShaveRt() ? findSym(op.getActShaveRt()) : nullptr;
    mlir::ArrayAttr actShaveStacksAttr = actShaveStacks.size() ? mlir::ArrayAttr::get(ctx, actShaveStacks) : nullptr;
    mlir::SymbolRefAttr dmaHwpBase = op.getDmaHwpBase() ? findSym(op.getDmaHwpBase()) : nullptr;
    mlir::SymbolRefAttr workpointCfg = op.getHwpWorkpointCfg() ? findSym(op.getHwpWorkpointCfg()) : nullptr;

    // TODO: E#100357
    mlir::FlatSymbolRefAttr managedMPISymRef;
    if (op.getWorkItemCount()) {
        llvm::SmallVector<llvm::SmallVector<mlir::Attribute>> managedDmas;
        for (auto tileDmas : llvm::enumerate(op.getDmaTasks())) {
            auto& newList = managedDmas.emplace_back();
            for (auto dma : llvm::enumerate(tileDmas.value())) {
                auto dmaTask = mlir::cast<VPUMI40XX::NNDMAOp>(dma.value().getDefiningOp());
                if (!dmaTask.isHardLinked())
                    continue;

                auto dmaName = findSym(dma.value());
                newList.push_back(dmaName);
            }
        }

        SmallVector<mlir::Attribute> managedDmasAttrVec;
        for (const auto& dma : managedDmas) {
            managedDmasAttrVec.push_back(mlir::ArrayAttr::get(ctx, dma));
        }
        mlir::ArrayAttr managedDmasAttr =
                managedDmasAttrVec.size() ? mlir::ArrayAttr::get(ctx, managedDmasAttrVec) : nullptr;

        auto managedMPISymName = mlir::StringAttr::get(rewriter.getContext(), symName.str() + std::string("_managed"));
        auto workItems = findSym(op.getWorkItemTasks());

        auto workItemCount = 0;
        if (op.getWorkItemCount().has_value()) {
            workItemCount = op.getWorkItemCount().value();
        }
        auto bootstrapTasksCount = 0;
        if (op.getBootstrapTasksCount().has_value()) {
            bootstrapTasksCount = op.getBootstrapTasksCount().value();
        }
        auto finalBarrierId = 0;
        if (op.getFinalBarrierId().has_value()) {
            finalBarrierId = op.getFinalBarrierId().value();
        }

        auto bootstrapWorkItemTasksCount = op.getBootsrapWorkItemsCount().value_or(0);

        mlir::SymbolRefAttr bootstrapItems = op.getBootstrapTasks() ? findSym(op.getBootstrapTasks()) : nullptr;
        auto managedMPI = rewriter.create<VPUASM::ManagedMappedInferenceOp>(
                op.getLoc(), managedMPISymName, managedDmasAttr, workItems, barrierTasksAttr, bootstrapItems,
                op.getDmaCountAttr(), workItemCount, op.getBarrierCount(), finalBarrierId, bootstrapTasksCount,
                bootstrapWorkItemTasksCount);

        managedMPISymRef = mlir::FlatSymbolRefAttr::get(managedMPISymName);
        rewriter.setInsertionPoint(managedMPI);
    }

    rewriter.create<VPUASM::MappedInferenceOp>(
            op.getLoc(), symName, dmasAttr, invariantTasksAttr, variantTasksAttr, actKernelRangesAttr,
            actKernelInvocationsAttr, mediaTasksAttr, barrierTasksAttr, actShaveRtAttr, actShaveStacksAttr,
            managedMPISymRef, op.getDmaCountAttr(), op.getInvariantCountAttr(), op.getVariantCountAttr(),
            op.getActKernelRangesCountAttr(), op.getActKernelInvocationsCountAttr(), op.getMediaCountAttr(),
            op.getBarrierCountAttr(), dmaHwpBase, workpointCfg);

    // Create MappedInferenceVersionOp as well
    rewriter.create<VPUASM::MappedInferenceVersionOp>(op.getLoc());

    // Also create PlatformInfoOp here, as it is related
    rewriter.create<VPUASM::PlatformInfoOp>(op.getLoc());

    rewriter.eraseOp(op);

    return mlir::success();
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
