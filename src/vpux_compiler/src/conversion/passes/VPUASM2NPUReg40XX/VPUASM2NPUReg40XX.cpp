//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp"
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/utils.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/VPU/utils/dry_run_utils.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/dialect/VPURegMapped/utils.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <npu_40xx_nnrt.hpp>

using namespace vpux;
using namespace vpux::VPURegMapped;
using namespace npu40xx;

namespace {

class BarrierRewriter final : public mlir::OpRewritePattern<VPUASM::ConfigureBarrierOp> {
public:
    BarrierRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUASM::ConfigureBarrierOp>(ctx), _log(log) {
        setDebugName("ConfigureBarrier_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::ConfigureBarrierOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult BarrierRewriter::matchAndRewrite(VPUASM::ConfigureBarrierOp origOp,
                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    // origOp.getNextSameId() is int64 with invalid barrier represented by -1 and a max
    // value of numeric_limits<uint32_t>::max() - 1
    // At this point it is cast to uint32 as required by the NNRuntime with invalid barrier
    // represented by numeric_limits<uint32_t>::max()
    auto regBarrierDescriptorAttr =
            vpux::VPURegMapped::getRegMappedAttributeWithValues<vpux::NPUReg40XX::RegMapped_VpuBarrierCountConfigType>(
                    rewriter, {
                                      {"next_same_id_",
                                       {{"next_same_id_", checked_cast_reg<NPUReg40XX::RegField_next_same_id_Type>(
                                                                  static_cast<uint32_t>(origOp.getNextSameId()))}}},
                                      {"producer_count_", {{"producer_count_", origOp.getProducerCount()}}},
                                      {"consumer_count_", {{"consumer_count_", origOp.getConsumerCount()}}},
                                      {"real_id_", {{"real_id_", origOp.getId()}}},
                              });
    rewriter.create<NPUReg40XX::ConfigureBarrierOp>(origOp->getLoc(), regBarrierDescriptorAttr);

    rewriter.eraseOp(origOp);

    return mlir::success();
}

class MappedInferenceRewriter final : public mlir::OpRewritePattern<VPUASM::MappedInferenceOp> {
public:
    MappedInferenceRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUASM::MappedInferenceOp>(ctx), _log(log) {
        setDebugName("MappedInference_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::MappedInferenceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MappedInferenceRewriter::matchAndRewrite(VPUASM::MappedInferenceOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto dmaCount = parseIntArrayOfArrayAttr<int64_t>(origOp.getDmaCount());

    mlir::SmallVector<int64_t> dmaCountDDR;
    mlir::SmallVector<int64_t> dmaCountCMX;
    dmaCountDDR.reserve(dmaCount.size());
    dmaCountCMX.reserve(dmaCount.size());

    for (size_t dmaTileIndex = 0; dmaTileIndex < dmaCount.size(); dmaTileIndex++) {
        VPUX_THROW_UNLESS(dmaCount[dmaTileIndex].size() == 2, "Unsupported number of DMA types - '{0}'",
                          dmaCount[dmaTileIndex].size());

        dmaCountDDR.push_back(dmaCount[dmaTileIndex][static_cast<size_t>(VPUMI40XX::DmaNnSrcType::DDR)]);
        dmaCountCMX.push_back(dmaCount[dmaTileIndex][static_cast<size_t>(VPUMI40XX::DmaNnSrcType::CMX_NN)]);
    }

    const auto dmaCountDDRAttr = getIntArrayAttr(origOp.getContext(), ArrayRef(dmaCountDDR));
    const auto dmaCountCMXAttr = getIntArrayAttr(origOp.getContext(), ArrayRef(dmaCountCMX));

    rewriter.create<NPUReg40XX::MappedInferenceOp>(origOp->getLoc(),                           //
                                                   origOp.getSymNameAttr(),                    //
                                                   dmaCountDDRAttr,                            //
                                                   dmaCountCMXAttr,                            //
                                                   origOp.getInvariantCountAttr(),             //
                                                   origOp.getVariantCountAttr(),               //
                                                   origOp.getActKernelRangesCountAttr(),       //
                                                   origOp.getActKernelInvocationsCountAttr(),  //
                                                   origOp.getMediaCountAttr(),                 //
                                                   origOp.getBarrierCountAttr(),               //
                                                   origOp.getActShaveRtAttr(),                 //
                                                   origOp.getActShaveStacksAttr(),             //
                                                   origOp.getDmaHwpBaseAttr(),                 //
                                                   origOp.getHwpWorkpointCfgAttr(),            //
                                                   origOp.getManagedMappedInferenceAttr());    //
    rewriter.eraseOp(origOp);

    return mlir::success();
}

//
// MappedInferenceVersion
//

class MappedInferenceVersionRewriter final : public mlir::OpRewritePattern<VPUASM::MappedInferenceVersionOp> {
public:
    MappedInferenceVersionRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUASM::MappedInferenceVersionOp>(ctx), _log(log) {
        setDebugName("MappedInferenceVersion_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::MappedInferenceVersionOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MappedInferenceVersionRewriter::matchAndRewrite(VPUASM::MappedInferenceVersionOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    rewriter.replaceOpWithNewOp<NPUReg40XX::MappedInferenceVersionOp>(origOp);
    return mlir::success();
}

class ManagedBarrierRewriter final : public mlir::OpRewritePattern<VPUASM::ManagedBarrierOp> {
public:
    ManagedBarrierRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUASM::ManagedBarrierOp>(ctx), _log(log) {
        setDebugName("ManagedBarrier_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::ManagedBarrierOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ManagedBarrierRewriter::matchAndRewrite(VPUASM::ManagedBarrierOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto workItemIdx = origOp.getWorkItemIdx();

    uint32_t workItemRegVal = 0;
    uint32_t enqueueCount = 0;

    if (workItemIdx.has_value()) {
        enqueueCount = origOp.getWorkItemCount();
        workItemRegVal = workItemIdx.value().getValue();
    }

    auto regBarrierDescriptorAttr =
            vpux::VPURegMapped::getRegMappedAttributeWithValues<vpux::NPUReg40XX::RegMapped_vpuTaskBarrierMapType>(
                    rewriter, {{"tb_next_same_id",
                                {{"tb_next_same_id", checked_cast_reg<NPUReg40XX::RegField_next_same_id_Type>(
                                                             static_cast<uint32_t>(origOp.getNextSameId()))}}},
                               {"tb_producer_count", {{"tb_producer_count", origOp.getProducerCount()}}},
                               {"tb_consumer_count", {{"tb_consumer_count", origOp.getConsumerCount()}}},
                               {"tb_real_id", {{"tb_real_id", origOp.getId()}}},
                               {"tb_work_item_idx",
                                {{"tb_work_item_idx",
                                  checked_cast_reg<NPUReg40XX::RegField_tb_work_item_idxType>(workItemRegVal)}}},
                               {"tb_enqueue_count",
                                {{"tb_enqueue_count",
                                  checked_cast_reg<NPUReg40XX::RegField_tb_enqueue_countType>(enqueueCount)}}}});

    rewriter.create<NPUReg40XX::ManagedBarrierOp>(origOp.getLoc(), regBarrierDescriptorAttr);
    rewriter.eraseOp(origOp);

    return mlir::success();
}

class ManagedMappedInferenceRewriter final : public mlir::OpRewritePattern<VPUASM::ManagedMappedInferenceOp> {
public:
    ManagedMappedInferenceRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUASM::ManagedMappedInferenceOp>(ctx), _log(log) {
        setDebugName("ManagedMappedInference_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::ManagedMappedInferenceOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    enum class DmaNnSrcType { DDR, CMX_NN, Count };
    Logger _log;
};

mlir::LogicalResult ManagedMappedInferenceRewriter::matchAndRewrite(VPUASM::ManagedMappedInferenceOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto dmaCount = parseIntArrayOfArrayAttr<int64_t>(origOp.getDmaCount());

    mlir::SmallVector<int64_t> dmaCountDDR;
    mlir::SmallVector<int64_t> dmaCountCMX;
    dmaCountDDR.reserve(dmaCount.size());
    dmaCountCMX.reserve(dmaCount.size());

    for (size_t dmaTileIndex = 0; dmaTileIndex < dmaCount.size(); dmaTileIndex++) {
        VPUX_THROW_UNLESS(dmaCount[dmaTileIndex].size() == 2, "Unsupported number of DMA types - '{0}'",
                          dmaCount[dmaTileIndex].size());

        dmaCountDDR.push_back(dmaCount[dmaTileIndex][static_cast<size_t>(VPUMI40XX::DmaNnSrcType::DDR)]);
        dmaCountCMX.push_back(dmaCount[dmaTileIndex][static_cast<size_t>(VPUMI40XX::DmaNnSrcType::CMX_NN)]);
    }

    const auto dmaCountDDRAttr = getIntArrayAttr(origOp.getContext(), ArrayRef(dmaCountDDR));
    const auto dmaCountCMXAttr = getIntArrayAttr(origOp.getContext(), ArrayRef(dmaCountCMX));

    rewriter.create<NPUReg40XX::ManagedMappedInferenceOp>(origOp->getLoc(),                    //
                                                          origOp.getSymNameAttr(),             //
                                                          origOp.getFinalBarrierId(),          //
                                                          dmaCountDDRAttr,                     //
                                                          dmaCountCMXAttr,                     //
                                                          origOp.getWorkItemsCount(),          //
                                                          origOp.getBarrierCount(),            //
                                                          origOp.getBootsrapWorkItemsCount(),  //
                                                          origOp.getBootstrapTasksCount(),     //
                                                          origOp.getActshvUsed(),              //
                                                          origOp.getDpuUsed(),                 //
                                                          origOp.getMediaUsed(),               //
                                                          origOp.getDmaFromDdrUsed(),          //
                                                          origOp.getDmaFromCmxUsed());         //
    rewriter.eraseOp(origOp);

    return mlir::success();
}

class WorkItemRewriter final : public mlir::OpRewritePattern<VPUASM::WorkItemOp> {
public:
    WorkItemRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUASM::WorkItemOp>(ctx), _log(log) {
        setDebugName("WorkItem_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::WorkItemOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult WorkItemRewriter::matchAndRewrite(VPUASM::WorkItemOp origOp,
                                                      mlir::PatternRewriter& rewriter) const {
    enum TaskType : uint8_t { DPU = 0, DMA, KERNEL, SYSTEM_MANAGEMENT, UNKNOWN = 255 };

    auto realTaskIndex = origOp.getRealTaskIndex();

    uint64_t descPtrOffset = 0;
    TaskType workItemType;

    switch (origOp.getTaskType()) {
    case VPURegMapped::TaskType::DPUVariant:
        workItemType = TaskType::DPU;
        descPtrOffset = static_cast<uint64_t>(1) << (realTaskIndex.getTileIdx() + NPUReg40XX::CMX_TILE_SELECT_OFFSET);
        break;
    case VPURegMapped::TaskType::DMA:
        workItemType = TaskType::DMA;
        break;
    case VPURegMapped::TaskType::ActKernelInvocation:
        workItemType = TaskType::KERNEL;
        descPtrOffset = static_cast<uint64_t>(1) << (realTaskIndex.getTileIdx() + NPUReg40XX::CMX_TILE_SELECT_OFFSET);
        break;
    default:
        return origOp.emitOpError("Invalid workItem task type");
        ;
    }

    auto regWorkItemDescriptorAttr =
            vpux::VPURegMapped::getRegMappedAttributeWithValues<vpux::NPUReg40XX::RegMapped_WorkItemType>(
                    rewriter, {
                                      {"desc_ptr", {{"desc_ptr", descPtrOffset}}},
                                      {"wi_type", {{"wi_type", workItemType}}},
                                      {"wi_unit", {{"wi_unit", realTaskIndex.getTileIdx()}}},
                                      {"wi_sub_unit", {{"wi_sub_unit", realTaskIndex.getListIdx()}}},

                              });

    rewriter.create<NPUReg40XX::WorkItemOp>(origOp.getLoc(), regWorkItemDescriptorAttr);
    rewriter.eraseOp(origOp);
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());
    return mlir::success();
}

//
// PlatformInfo
//

class PlatformInfoRewriter final : public mlir::OpRewritePattern<VPUASM::PlatformInfoOp> {
public:
    PlatformInfoRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUASM::PlatformInfoOp>(ctx), _log(log) {
        setDebugName("PlatformInfo_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::PlatformInfoOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult PlatformInfoRewriter::matchAndRewrite(VPUASM::PlatformInfoOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    rewriter.replaceOpWithNewOp<VPUASM::PlatformInfoOp>(origOp, VPU::ArchKind::NPU40XX);
    return mlir::success();
}

//
// ConvertVPUASM2NPUReg40XXPass
//

class ConvertVPUASM2NPUReg40XXPass final : public ConvertVPUASM2NPUReg40XXBase<ConvertVPUASM2NPUReg40XXPass> {
public:
    explicit ConvertVPUASM2NPUReg40XXPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnModule() final;
};

void ConvertVPUASM2NPUReg40XXPass::safeRunOnModule() {
    auto moduleOp = getOperation();
    auto& ctx = getContext();
    mlir::func::FuncOp netFunc;
    IE::CNNNetworkOp cnnOp;

    IE::CNNNetworkOp::getFromModule(moduleOp, cnnOp, netFunc);

    mlir::ConversionTarget target(ctx);

    target.addLegalDialect<ELF::ELFDialect>();
    target.addLegalDialect<NPUReg40XX::NPUReg40XXDialect>();
    target.addLegalDialect<VPUASM::VPUASMDialect>();

    target.addIllegalOp<VPUASM::ConfigureBarrierOp>();
    target.addIllegalOp<VPUASM::MappedInferenceOp>();
    target.addIllegalOp<VPUASM::MappedInferenceVersionOp>();
    target.addIllegalOp<VPUASM::ManagedBarrierOp>();
    target.addIllegalOp<VPUASM::WorkItemOp>();
    target.addIllegalOp<VPUASM::ManagedMappedInferenceOp>();

    target.addDynamicallyLegalOp<VPUASM::PlatformInfoOp>([&](VPUASM::PlatformInfoOp op) {
        return op.getArchKind() != VPU::ArchKind::UNKNOWN;
    });

    mlir::RewritePatternSet patterns(&ctx);

    patterns.add<BarrierRewriter>(&ctx, _log);
    patterns.add<MappedInferenceRewriter>(&ctx, _log);
    patterns.add<ManagedBarrierRewriter>(&ctx, _log);
    patterns.add<ManagedMappedInferenceRewriter>(&ctx, _log);
    patterns.add<WorkItemRewriter>(&ctx, _log);
    patterns.add<MappedInferenceVersionRewriter>(&ctx, _log);
    patterns.add<PlatformInfoRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(netFunc, target, std::move(patterns)))) {
        signalPassFailure();
    }

    return;
}

}  // namespace

//
// createConvertVPUASM2NPUReg40XXPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertVPUASM2NPUReg40XXPass(Logger log) {
    return std::make_unique<ConvertVPUASM2NPUReg40XXPass>(log);
}
