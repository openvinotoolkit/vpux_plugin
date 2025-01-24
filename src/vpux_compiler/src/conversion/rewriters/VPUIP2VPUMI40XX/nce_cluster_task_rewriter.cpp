//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/conversion/rewriters/VPUIP2VPUMI40XX/nce_cluster_task_rewriter.hpp"
#include "vpux/compiler/conversion/passes/VPUIP2VPUMI40XX/buffer_conversion.hpp"

#include "vpux/compiler/dialect/IE/utils/resources.hpp"

#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace {

// E#145191:
// even though backend shouldn't really validate VPUIP IR
// there seem to be no such checks higher on the stack

template <class RangeT, class Property>
void checkIfDPUTasksAreDifferent(RangeT dpuTasks, Property&& functor) {
    const auto difference =
            std::adjacent_find(std::begin(dpuTasks), std::end(dpuTasks), [&functor](auto lhs, auto rhs) {
                return functor(lhs) != functor(rhs);
            });
    if (difference == dpuTasks.end()) {
        return;
    }

    auto lhs = *difference;
    auto rhs = *std::next(difference);
    VPUX_THROW("DPU tasks {} and {} from the same NCEClusterTaskOp are different: {} vs {}", lhs, rhs, functor(lhs),
               functor(rhs));
}

template <class RangeT>
void checkAllDPUTasksHaveTheSameMode(RangeT dpuTasks) {
    checkIfDPUTasksAreDifferent(std::move(dpuTasks), [](auto dpuTask) {
        return dpuTask.getMpeMode();
    });
}

template <class RangeT>
void checkAllDPUTasksHaveTheSameClusterID(RangeT dpuTasks) {
    checkIfDPUTasksAreDifferent(dpuTasks, [](auto dpuTask) {
        const auto maybeClusterID = dpuTask.getClusterId();
        assert(maybeClusterID.has_value());
        return maybeClusterID.value();
    });
}

void minimizeWorkloadSize(mlir::OpBuilder& builder, VPUMI40XX::DPUVariantOp variant, VPUIP::NCETaskType taskType,
                          std::optional<mlir::ArrayAttr> kernelSize) {
    auto newInEnd = parseIntArrayAttr<int64_t>(variant.getInStart());
    auto newOutEnd = parseIntArrayAttr<int64_t>(variant.getStart());

    int64_t minX = 1, minY = 1;
    if (kernelSize.has_value()) {
        const auto kernelSizeArray = parseIntArrayAttr<int64_t>(kernelSize.value());
        minX = kernelSizeArray[1];
        minY = kernelSizeArray[0];
    }

    const auto pad = variant.getPad();
    minX -= pad.getLeft().getInt();
    minX -= pad.getRight().getInt();
    minY -= pad.getTop().getInt();
    minY -= pad.getBottom().getInt();

    newInEnd[0] += minX - 1;
    newInEnd[1] += minY - 1;

    if (taskType == VPUIP::NCETaskType::ELTWISE) {
        // Eltwise doesn't support splitting over Z, so we leave it the same
        newInEnd[2] = parseIntArrayAttr<int64_t>(variant.getInEnd())[2];
        newOutEnd[2] = parseIntArrayAttr<int64_t>(variant.getEnd())[2];
    } else {
        constexpr int minWorkloadZ = 16;
        newInEnd[2] += minWorkloadZ - 1;
        newOutEnd[2] += minWorkloadZ - 1;
    }

    variant.setInEndAttr(getIntArrayAttr(builder, newInEnd));
    variant.setEndAttr(getIntArrayAttr(builder, newOutEnd));
}

}  // namespace

namespace vpux::vpuip2vpumi40xx {

mlir::LogicalResult NCEClusterTaskRewriter::matchAndRewrite(VPUIP::NCEClusterTaskOp origOp, OpAdaptor adaptor,
                                                            mlir::ConversionPatternRewriter& rewriter) const {
    auto ctx = origOp.getContext();

    auto dpuTasks = adaptor.getVariants().getOps<VPUIP::DPUTaskOp>();
    assert(!dpuTasks.empty());

    checkAllDPUTasksHaveTheSameMode(dpuTasks);
    // E#145191: ambiguous check, requires clarification
    // checkAllDPUTasksHaveTheSameClusterID(dpuTasks);
    // const auto tileIndex = (*dpuTasks.begin()).getClusterId().value();

    // E#145194: refactor to get cluster id easier
    uint32_t tileIndex = 0;
    if ((*dpuTasks.begin()).getClusterId().has_value()) {
        tileIndex = (*dpuTasks.begin()).getClusterId().value();
    } else {
        auto bufferOp = mlir::cast<VPURT::DeclareBufferOp>(origOp.getInput().getDefiningOp());
        if (bufferOp.getSection() == VPURT::BufferSection::CMX_NN) {
            if (bufferOp.getSectionIndex().has_value() && !bufferOp.getSectionIndex().value().empty()) {
                auto tiles = parseIntArrayAttr<uint8_t>(bufferOp.getSectionIndex().value());
                tileIndex = *std::min_element(tiles.begin(), tiles.end());
            }
        }
    }

    const auto mpeModeAttr = (*dpuTasks.begin()).getMpeModeAttr();

    const auto indexWithOnlyTileSet = VPURegMapped::IndexType::get(ctx, tileIndex, 0, 0);
    const auto zeroUI64Attr = mlir::IntegerAttr::get(getUInt64Type(ctx), 0);

    auto weights = convertOrExtractBuffer(rewriter, adaptor.getWeights(), tileIndex);
    auto weightTable = convertOrExtractBuffer(rewriter, adaptor.getWeightTable(), tileIndex);
    auto weightTableDataPtr = convertOrExtractBuffer(rewriter, adaptor.getWeightTableDataPtr(), tileIndex);
    auto weightTableSpPtr = convertOrExtractBuffer(rewriter, adaptor.getWeightTableSpPtr(), tileIndex);
    auto weightTableScale = convertOrExtractBuffer(rewriter, adaptor.getWeightTableScale(), tileIndex);
    auto weightTableBias = convertOrExtractBuffer(rewriter, adaptor.getWeightTableBias(), tileIndex);
    auto weightZeroPoints = convertOrExtractBuffer(rewriter, adaptor.getWeightZeroPoints(), tileIndex);
    auto sprLookupTable = convertOrExtractBuffer(rewriter, adaptor.getSprLookupTable(), tileIndex);
    auto taskTypeAttr = adaptor.getTaskTypeAttr();

    auto invariant = rewriter.create<VPUMI40XX::DPUInvariantOp>(
            origOp.getLoc(), indexWithOnlyTileSet,
            nullptr,  // taskLocation
            nullptr,  // previousInvariant
            convertOrExtractBuffer(rewriter, adaptor.getInput(), tileIndex),
            convertOrExtractBuffer(rewriter, adaptor.getInputSparsityMap(), tileIndex),
            convertOrExtractBuffer(rewriter, adaptor.getInputStorageElementTable(), tileIndex), weights,
            convertOrExtractBuffer(rewriter, adaptor.getWeightsSparsityMap(), tileIndex), weightTable,
            weightTableDataPtr, weightTableSpPtr, weightTableScale, weightTableBias, weightZeroPoints, sprLookupTable,
            convertOrUnrollBuffer(rewriter, adaptor.getOutputBuff()),
            convertOrUnrollBuffer(rewriter, adaptor.getOutputSparsityMapBuff()), adaptor.getProfilingData(),
            adaptor.getMaxPerXy(), adaptor.getMinPerXy(), adaptor.getMinMaxPerTensor(), taskTypeAttr,
            adaptor.getEltwiseTypeAttr(), mpeModeAttr, adaptor.getMpeEngineAttr(), adaptor.getKernelSizeAttr(),
            adaptor.getKernelStridesAttr(), adaptor.getKernelPaddingAttr(), adaptor.getIsContinuedAttr(),
            adaptor.getCmSpPatternAttr(), adaptor.getInputChannelsCompressionAttr(),
            adaptor.getIsZeroOffsetWeightsTableAttr(), adaptor.getOutChannelOffsetAttr(), adaptor.getIsSuperdenseAttr(),
            adaptor.getIsInplaceAttr(), adaptor.getInputSeSizeAttr(), adaptor.getOutputSeSizeAttr(),
            adaptor.getIsPermuteQuantizeAttr(), adaptor.getIsSmallKernelOptimizedAttr(),
            adaptor.getProfilingMetadataAttr(),
            mlir::ValueRange(),  // waitBarriers
            mlir::ValueRange(),  // updateBarriers
            zeroUI64Attr,        // startAfter
            zeroUI64Attr,        // cleanAfter
            nullptr              // enqueueBarrier
    );

    auto createVPUMI40XXVariant = [&](auto dpuTask, mlir::UnitAttr lutRead, mlir::UnitAttr forceInvRead) {
        return rewriter.create<VPUMI40XX::DPUVariantOp>(
                dpuTask.getLoc(), indexWithOnlyTileSet,
                nullptr,  // taskLocation
                nullptr,  // previousVariant
                invariant.getResult(), weights, weightTable, weightTableDataPtr, weightTableSpPtr, weightTableScale,
                weightTableBias, weightZeroPoints, taskTypeAttr, dpuTask.getInStartAttr(), dpuTask.getInEndAttr(),
                dpuTask.getOutStartAttr(), dpuTask.getOutEndAttr(), dpuTask.getPadAttr(), mpeModeAttr,
                mlir::IntegerAttr::get(getUInt64Type(ctx), tileIndex), dpuTask.getHaloRegionsAttr(),
                dpuTask.getWorkloadIdAttr(), lutRead, forceInvRead);
    };

    // As sprLUT is prefetched before all the barriers for the DPU are produced (see DPU FSM diagram in
    // HAS), We need some way to make sure that once we read sprLUT, the DMA task that brings it is
    // finished. This is done by adding an additional dummy DPU variant, which exists purely for waiting
    // until all the barriers are produced.
    if (sprLookupTable) {
        auto dummyVariant = createVPUMI40XXVariant(*dpuTasks.begin(), /*lutRead=*/nullptr, /*forceInvRead=*/nullptr);
        minimizeWorkloadSize(rewriter, dummyVariant, invariant.getNceTaskType(), invariant.getKernelSize());
    }

    for (auto dpuTask : dpuTasks) {
        // For the first variant that goes after the dummy one, two additional registers are set:
        // - lut_read enables the read of sprLUT (it can only be done once per invariant as other variants will
        // just reuse the loaded one)
        // - force_inv_read forces re-read of the Invariant. sprLUT read is triggered only as a part of Invariant
        // read (see DPU FSM diagram in HAS) and Invariant read may be skipped if it's already loaded. As Dummy
        // DPU variant loads Invariant for this workload, without it read of sprLUT may be skipped as well, no
        // matter what we set in readLut.
        createVPUMI40XXVariant(dpuTask, /*lutRead=*/sprLookupTable ? mlir::UnitAttr::get(ctx) : nullptr,
                               /*forceInvRead=*/sprLookupTable ? mlir::UnitAttr::get(ctx) : nullptr);
        sprLookupTable = nullptr;
    }

    {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        auto& invariantPPERegion = invariant.getPpe();
        invariantPPERegion.emplaceBlock();
        rewriter.setInsertionPointToEnd(&invariantPPERegion.front());

        for (auto ppe : origOp.getPpe().getOps<VPUIP::PPETaskOp>()) {
            rewriter.create<VPUMI40XX::PPETaskOp>(ppe.getLoc(), ppe->getResultTypes(), ppe->getOperands(),
                                                  ppe->getAttrDictionary().getValue());
        }
    }

    rewriter.eraseOp(origOp);
    return mlir::success();
}

}  // namespace vpux::vpuip2vpumi40xx
