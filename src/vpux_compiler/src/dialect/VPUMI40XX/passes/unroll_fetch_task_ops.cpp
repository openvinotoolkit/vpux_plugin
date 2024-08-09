//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/ops.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <npu_40xx_nnrt.hpp>

using namespace vpux;

namespace {

class RewriteEnqueueToDma final : public mlir::OpRewritePattern<VPURegMapped::FetchTaskOp> {
public:
    RewriteEnqueueToDma(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPURegMapped::FetchTaskOp>(ctx), _log(log) {
        setDebugName("EnuqeueOpRewriter");
    }

    mlir::LogicalResult matchAndRewrite(VPURegMapped::FetchTaskOp FetchTaskOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    int64_t getTaskSize(VPURegMapped::TaskType taskType) const;
    Logger _log;
};

int64_t RewriteEnqueueToDma::getTaskSize(VPURegMapped::TaskType taskType) const {
    switch (taskType) {
    case VPURegMapped::TaskType::DPUInvariant:
        return sizeof(npu40xx::nn_public::VpuDPUInvariant);
        break;
    case VPURegMapped::TaskType::DPUVariant:
        return sizeof(npu40xx::nn_public::VpuDPUVariant);
        break;
    case VPURegMapped::TaskType::ActKernelInvocation:
        return sizeof(npu40xx::nn_public::VpuActKernelInvocation);
        break;
    case VPURegMapped::TaskType::ActKernelRange:
        return sizeof(npu40xx::nn_public::VpuActKernelRange);
        break;
    default:
        VPUX_THROW("Unknow Task Type {0}", taskType);
        break;
    }
}

mlir::LogicalResult RewriteEnqueueToDma::matchAndRewrite(VPURegMapped::FetchTaskOp FetchTaskOp,
                                                         mlir::PatternRewriter& rewriter) const {
    auto ctx = getContext();
    auto primaryTaskOpStart = mlir::cast<VPURegMapped::TaskOpInterface>(FetchTaskOp.getPrimaryStart().getDefiningOp());
    auto primaryTaskOpEnd = mlir::cast<VPURegMapped::TaskOpInterface>(FetchTaskOp.getPrimaryEnd().getDefiningOp());
    auto primarySize = getTaskSize(primaryTaskOpStart.getTaskType());

    auto secondaryTaskOpStart =
            mlir::cast<VPURegMapped::TaskOpInterface>(FetchTaskOp.getSecondaryStart().getDefiningOp());
    auto secondaryTaskOpEnd = mlir::cast<VPURegMapped::TaskOpInterface>(FetchTaskOp.getSecondaryEnd().getDefiningOp());
    auto secondarySize = getTaskSize(secondaryTaskOpStart.getTaskType());

    // zero based indexes distance need to be incremented by 1
    int64_t primaryCount =
            primaryTaskOpEnd.getIndexType().getValue() - primaryTaskOpStart.getIndexType().getValue() + 1;
    int64_t secondaryCount =
            secondaryTaskOpEnd.getIndexType().getValue() - secondaryTaskOpStart.getIndexType().getValue() + 1;

    const auto memSpaceCMX = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN),
                                                          primaryTaskOpStart.getIndexType().getTileIdx());

    auto primaryMemrefDDR = mlir::MemRefType::get({primaryCount, primarySize}, rewriter.getIntegerType(8, false));
    auto primaryMemrefCMX =
            primaryMemrefDDR.cast<NDTypeInterface>().changeMemSpace(memSpaceCMX).cast<mlir::MemRefType>();

    auto secondaryMemrefDDR = mlir::MemRefType::get({secondaryCount, secondarySize}, rewriter.getIntegerType(8, false));
    auto secondaryMemrefCMX =
            secondaryMemrefDDR.cast<NDTypeInterface>().changeMemSpace(memSpaceCMX).cast<mlir::MemRefType>();

    auto primaryTaskView = rewriter.create<VPURegMapped::ViewTaskRangeOp>(
            FetchTaskOp.getLoc(), primaryMemrefDDR, FetchTaskOp.getPrimaryStart(), FetchTaskOp.getPrimaryEnd());

    auto primaryTaskLocationsView = rewriter.create<VPURegMapped::ViewTaskRangeOp>(
            FetchTaskOp.getLoc(), primaryMemrefCMX, primaryTaskOpStart.getTaskLocation(),
            primaryTaskOpEnd.getTaskLocation());

    auto secondaryTaskView = rewriter.create<VPURegMapped::ViewTaskRangeOp>(
            FetchTaskOp.getLoc(), secondaryMemrefDDR, FetchTaskOp.getSecondaryStart(), FetchTaskOp.getSecondaryEnd());

    auto secondaryTaskLocationsView = rewriter.create<VPURegMapped::ViewTaskRangeOp>(
            FetchTaskOp.getLoc(), secondaryMemrefCMX, secondaryTaskOpStart.getTaskLocation(),
            secondaryTaskOpEnd.getTaskLocation());

    auto primaryDma = rewriter.create<VPUMI40XX::NNDMAOp>(
            FetchTaskOp.getLoc(), FetchTaskOp.getIndexType(),
            nullptr,  // for now it's assumed that with WLM DMA's don't have a taskLocation
            primaryTaskView.getResult(), mlir::ValueRange({primaryTaskLocationsView.getResult()}),
            FetchTaskOp.getPreviousTask(),               // inherit the previous
            mlir::ValueRange({}), mlir::ValueRange({}),  // NO BARRIERS for these DMA's for
            0, 0,                                        // start_after, clean_after fields have no meaning with WLM
            true, true, false,                           // is_out_of_rder  and is_critical, enable_msc
            0,                                           // port has no meaning
            VPUIP::DMAAccMode::DISABLE,
            nullptr,  // dma_transaction
            nullptr,  // no descriptor attr required
            nullptr,  // no act_compression_sparsity_map required
            nullptr,  // no descriptor attr required
            nullptr,  // dma_hwp_id 0 s nullptr
            nullptr,  //  profilingMetadata
            0,        // allow_different_in_out_shapes
            nullptr   // indices
    );
    auto secondaryDma = rewriter.create<VPUMI40XX::NNDMAOp>(
            FetchTaskOp.getLoc(), FetchTaskOp.getIndexType(),
            nullptr,  // for now it's assumed that with WLM DMA's don't have a taskLocation
            secondaryTaskView.getResult(), mlir::ValueRange({secondaryTaskLocationsView.getResult()}),
            primaryDma.getResult(), mlir::ValueRange({}), mlir::ValueRange({}),  // NO BARRIERS for these DMA's for
            0, 0,               // start_after, clean_after fields have no meaning with WLM
            true, true, false,  // is_out_of_rder  and is_critical, enable_msc
            0,                  // port has no meaning
            VPUIP::DMAAccMode::DISABLE,
            nullptr,  // dma_transaction
            nullptr,  // no descriptor attr required
            nullptr,  // no act_compression_sparsity_map required
            nullptr,  // no descriptor attr required
            nullptr,  // dma_hwp_id 0 s nullptr
            nullptr,  // profilingMetadata
            0,        // allow_different_in_out_shapes
            nullptr   // indices
    );

    // the use of mapped inference is to be replaced with the FIRST dma.
    // the rest of the DMA's are to be replaced with the SECOND dma
    rewriter.replaceOpWithIf(FetchTaskOp.getOperation(), mlir::ValueRange(primaryDma.getResult()),
                             [](mlir::OpOperand& operand) {
                                 return mlir::isa<VPUMI40XX::MappedInferenceOp>(operand.getOwner()) ||
                                        mlir::isa<VPUMI40XX::OpRanges>(operand.getOwner());
                             });
    rewriter.replaceOpWithIf(FetchTaskOp.getOperation(), mlir::ValueRange(secondaryDma.getResult()),
                             [](mlir::OpOperand& operand) {
                                 return !mlir::isa<VPUMI40XX::MappedInferenceOp>(operand.getOwner());
                             });

    rewriter.eraseOp(FetchTaskOp.getOperation());

    return mlir::success();
}

class UnrollFetchTaskOpsPass : public VPUMI40XX::UnrollFetchTaskOpsBase<UnrollFetchTaskOpsPass> {
public:
    explicit UnrollFetchTaskOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollFetchTaskOpsPass::safeRunOnFunc() {
    auto netFunc = getOperation();
    auto ctx = &getContext();

    mlir::RewritePatternSet patterns(ctx);
    patterns.add<RewriteEnqueueToDma>(ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(netFunc, std::move(patterns),
                                                        vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }

    auto parentModule = netFunc.getOperation()->getParentOfType<mlir::ModuleOp>();
    const auto tilesCount = IE::getTileExecutor(parentModule).getCount();

    auto mpi = VPUMI40XX::getMPI(netFunc);

    // auto builder = mlir::OpBuilder(mpi);
    const size_t DMA_DDR2CMX_LISTIDX = 0;

    for (int64_t tileIdx = 0; tileIdx < tilesCount; tileIdx++) {
        auto listHead = mpi.getListHead(VPURegMapped::TaskType::DMA, tileIdx, DMA_DDR2CMX_LISTIDX);
        if (!listHead)
            continue;

        auto newCount = VPUMI40XX::reindexList(mlir::cast<VPURegMapped::TaskOpInterface>(listHead.getDefiningOp()));

        auto dmaCount = parseIntArrayOfArrayAttr<int64_t>(mpi.getDmaCount());
        dmaCount[tileIdx][DMA_DDR2CMX_LISTIDX] = newCount;
        mpi.setDmaCountAttr(getIntArrayOfArray(ctx, dmaCount));
    }

    return;
}

}  // namespace

//
// createUnrollFetchTaskOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUMI40XX::createUnrollFetchTaskOpsPass(Logger log) {
    return std::make_unique<UnrollFetchTaskOpsPass>(log);
}
