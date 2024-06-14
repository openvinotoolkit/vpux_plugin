//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPURT/transforms/passes.hpp"

#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"

#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

void vpux::VPUIP::arch40xx::buildMemoryAllocationPipeline(mlir::OpPassManager& pm,
                                                          const VPUIP::arch40xx::MemoryAllocationOptions& options,
                                                          Logger log) {
    pm.addPass(VPUIP::createFeasibleAllocationPass(
            VPU::getMemKind<VPU::MemoryKind::CMX_NN>, VPU::getMemKind<VPU::MemoryKind::DDR>, options.linearizeSchedule,
            options.enablePipelining, options.enablePrefetching, options.optimizeFragmentation,
            options.optimizeDynamicSpilling, log));

    if (options.enableCompressActivationSpill) {
        pm.addPass(VPUIP::createAdjustSpillSizePass(log));
    }

    if (options.enableGroupAsyncExecuteOps) {
        pm.addPass(VPUIP::createGroupAsyncExecuteOpsPass(log));
    }

    pm.addPass(VPUIP::createQueryArgsAllocationAnalysisPass());
    pm.addPass(VPUIP::createStaticAllocationPass(VPU::getMemKind<VPU::MemoryKind::DDR>, log));
    pm.addPass(VPUIP::createCollectUsedMemoryPass());
}

void vpux::VPUIP::arch40xx::buildDefaultHWPipeline(mlir::OpPassManager& pm,
                                                   const VPUIP::arch40xx::DefaultHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(VPUIP::createTileActShaveKernelTaskPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    if (options.enableOpsAsDMA) {
        pm.addPass(VPUIP::createWrapWithPermuteAsNNDMAPass(log));
    }
    pm.addPass(VPUIP::createConvertExpandPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPUIP::createConvertEltwiseToInPlacePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // Level 2 : Abstract RunTime

    pm.addPass(VPUIP::createSetMemorySpacePass(VPU::getMemKind<VPU::MemoryKind::DDR>, log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableSEPtrsOperations || options.enableExperimentalSEPtrsOperations) {
        pm.addPass(VPUIP::createMoveSubViewBeforeSparseBufferPass(log));
        pm.addPass(VPUIP::createComputeSEBasePtrsPass(log));
        pm.addPass(VPUIP::createConvertSETablesToConstantsPass(log));
    }
    if (options.enableWeightsSparsity) {
        pm.addPass(VPUIP::createPropagateSparsityCompressionPass(log));
    }
    if (options.enableWeightsSparsity || VPU::isActSparsityEnabled(options.enableActivationSparsity)) {
        pm.addPass(VPUIP::createUngroupSparseBuffersPass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
    }

    if (options.enableOptimizeCopies) {
        pm.addPass(VPUIP::createMovePureViewOpBeforeCopyPass(log));
        pm.addPass(VPUIP::createOptimizeCopiesPass(log));
        pm.addPass(VPUIP::createOptimizeConcatViewCopiesPass(log));
        pm.addPass(VPUIP::createFuseDDRCopiesIntoConcats(log));
        pm.addPass(VPUIP::createOptimizeParallelCopiesPass(options.enableOptimizeConstCopies, log));
        if (options.enableOpsAsDMA) {
            pm.addPass(VPUIP::createMovePureViewOpBeforeCopyPass(log));
            pm.addPass(VPUIP::createWrapWithPermuteAsNNDMAPass(log));
        }
    }

    pm.addPass(VPUIP::createInsertCopyForEltwiseInPlaceInputPass(log));
    pm.addPass(VPUIP::arch40xx::createOptimizeConvertDMAOpPass(log));

    if (options.enableOpsAsDMA) {
        pm.addPass(VPUIP::createConvertToDMAPass(log));
    }
    pm.addPass(VPUIP::createCopyOpTilingPass(log));

    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(VPUIP::createUnwrapClusterTilingPass(log));
    pm.addPass(VPUIP::createConvWeightsCompressionPass(log));

    if (VPU::isActSparsityEnabled(options.enableActivationSparsity)) {
        pm.addPass(VPUIP::createComputeSESizesPass(/*onlyInputsConcatOverC=*/true, log));
    }

    if (options.enableConstantFusion) {
        pm.addPass(VPUIP::createFuseConstantsPass(log));
    }

    pm.addPass(VPUIP::createSwizzlingPass(options.enableWeightsSwizzling, options.enableActivationSwizzling, log));

    pm.addPass(VPUIP::createConvertTransferOpsToDMAsPass(log));

    if (options.enableProfiling && options.enableDPUProfiling) {
        pm.addPass(VPUIP::createDPUProfilingPass(VPU::getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }

    if (options.enableProfiling && options.enableSWProfiling) {
        pm.addPass(VPUIP::createActShaveProfilingPass(VPU::getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }

    VPUIP::buildAsyncSchedulingPipeline(pm, log);

    if (options.enableCompressActivationSpill) {
        pm.addPass(VPUIP::createCompressDmaReserveMemPass(log));
    }

    pm.addPass(VPUIP::createDMATaskProfilingReserveMemPass(log));

    if (options.enableSWKernelPrefetchingReserveMem) {
        pm.addPass(VPUIP::createSWKernelPrefetchingReserveMemPass(log));
    }

    pm.addPass(VPUIP::createCalculateAsyncRegionCycleCostPass(log));

    VPUIP::arch40xx::buildMemoryAllocationPipeline(pm, VPUIP::arch40xx::MemoryAllocationOptions(options), log);

    pm.addPass(VPUIP::createOptimizeAsyncDepsPass(log));

    // Handle WeightsTable, which requires statically allocated memory
    pm.addPass(VPUIP::createPatchWeightsTablePass(log));

    pm.addPass(VPUIP::arch37xx::createAddSwKernelCacheHandlingOpsPass(log));

    VPUIP::buildHardwareAdaptationPipeline(pm, log);

    // Level 1 : VPU RunTime
    pm.addPass(VPUIP::createUnrollSwKernelPass(log));

    pm.addPass(VPUIP::arch40xx::createUnrollClusterTilingPass(log));
    pm.addPass(VPUIP::createNNDMATilingPass(log));
    pm.addPass(VPUIP::createSegmentHalosPass(log));
    pm.addPass(VPUIP::arch40xx::createComputeHaloRegionForDPUTaskOpPass(log));

    if (options.enableWeightsSparsity) {
        pm.addPass(VPUIP::createFlattenSparseWeightsTypesPass(log));
    }
    if (VPU::isActSparsityEnabled(options.enableActivationSparsity) || options.enableSEPtrsOperations ||
        options.enableExperimentalSEPtrsOperations) {
        pm.addPass(VPUIP::createComputeSESizesPass(/*onlyInputsConcatOverC=*/false, log));
    }
    if (options.enableSEPtrsOperations || options.enableExperimentalSEPtrsOperations) {
        pm.addPass(VPUIP::createAdjustInputDataForExplicitSETablePass(log));
    }

    pm.addPass(VPUIP::createUnrollDepthToSpaceDMAPass(log));
    pm.addPass(VPUIP::createUnrollSpaceToDepthDMAPass(log));
    pm.addPass(VPUIP::createUnrollPermuteToNNDMAPass(log));

    pm.addPass(VPUIP::createUnrollUpsamplingDMAPass(log));
    pm.addPass(VPUIP::createUnrollExpandDMAPass(log));
    pm.addPass(VPUIP::createUnrollPerAxisTileDMAPass(log));

    pm.addPass(Const::createApplySwizzlingPass());
    pm.addPass(VPUIP::createResolveDMAWithSwizzlingPass(log));

    if (options.enableCompressWeightsBTC) {
        pm.addPass(VPUIP::createCompressWeightsBTCPass(log));
    }

    pm.addPass(VPUIP::arch40xx::createSplitDMAToBalanceLoadPass(log));

    if (options.enableCompressActivationSpill) {
        pm.addPass(VPUIP::arch40xx::createCompressSpillDmaPass(log));
    }

    if (options.enableFunctionOutlining) {
        if (options.enableBarrierSchedWithFunctionOutlining) {
            pm.addPass(VPURT::arch40xx::createInsertSyncTasksPass(log));
        } else {
            pm.addPass(mlir::createInlinerPass());
        }
    }

    if (options.enableControlGraphSplit) {
        pm.addPass(VPURT::createSplitControlGraphPass(options.controlGraphSplitBlockSize, log));
    }

    if (!options.linearizeSchedule) {
        pm.addPass(VPUIP::createDMABarrierOptimizationPass(log));
    }

    if (options.enableSimpleSchedule) {
        pm.addPass(VPURT::createSimplifySchedulePass(options.shareWaitAndUpdateBarriers,
                                                     options.reduceParallelControlFlows, log));
    }

    auto dpuDryRunMode = VPU::getDPUDryRunMode(options.dpuDryRun);
    if (dpuDryRunMode == VPU::DPUDryRunMode::STRIP || options.shaveDryRun == true) {
        pm.addPass(VPUIP::arch40xx::createComputeTaskStrippingPass(log, dpuDryRunMode, options.shaveDryRun));
    }

    VPURT::buildBarrierLegalizationPipeline(pm, log);

    if (options.enableFunctionOutlining && options.enableBarrierSchedWithFunctionOutlining) {
        pm.addPass(mlir::createInlinerPass());
        pm.addPass(VPURT::arch40xx::createOptimizeSyncTasksPass(log));
    }

    if (options.enableStartBarrier) {
        pm.addPass(VPUIP::arch40xx::createAddStartBarrierPass(log));
    }
    if (options.enableFinalBarrier) {
        pm.addPass(VPURT::arch37xx::createAddFinalBarrierPass(log));
    }

    pm.addPass(VPURT::arch37xx::createAddUpdateBarrierForSwKernelsPass(log));

    pm.addPass(VPUIP::arch40xx::createDMAOutOfOrderOptimizationPass(log));

    if (options.enableProfiling) {
        if (options.enableDMAProfiling) {
            pm.addPass(VPUIP::arch40xx::createDMATaskProfilingHwDdrPass(log));
        }
        if (options.enableDPUProfiling) {
            pm.addPass(VPUIP::arch40xx::createConstantDpuProfHwpBasePass(log));
        }
        pm.addPass(VPUIP::createCaptureWorkpointPass(log));
        pm.addPass(VPUIP::createGroupProfilingBuffersPass(log));
        pm.addPass(createMoveDeclarationsToTopPass(log));
    }

    pm.addPass(VPURT::createAssignPhysicalBarriersPass(log));
    pm.addPass(VPURT::createBarrierSimulationPass(log));
    mlir::OpPassManager::Nesting nesting = pm.getNesting();
    pm.setNesting(mlir::OpPassManager::Nesting::Explicit);
    mlir::OpPassManager& constPm = pm.nest<mlir::func::FuncOp>().nest<Const::DeclareOp>();
    constPm.addPass(Const::createConstantFoldingPass());
    pm.setNesting(nesting);

    if (options.enableActivityFactor || options.enableScheduleTrace) {
        pm.addPass(VPURT::createInferenceExecutionAnalysisPass(options.scheduleTraceFile, options.enableScheduleTrace,
                                                               options.enableActivityFactor, log));
    }
    if (options.enableDumpTaskStats) {
        // Force logging if dump-task-stats was enabled explicitly on the command line
        pm.addPass(VPUIP::createDumpStatisticsOfTaskOpsPass(
                log, options.enableDumpTaskStats.hasValue() && options.enableDumpTaskStats));
    }
}

void vpux::VPUIP::arch40xx::registerVPUIPPipelines() {
    mlir::PassPipelineRegistration<VPUIP::arch40xx::MemoryAllocationOptions>(
            "memory-allocation", "Memory Allocation",
            [](mlir::OpPassManager& pm, const VPUIP::arch40xx::MemoryAllocationOptions& options) {
                VPUIP::arch40xx::buildMemoryAllocationPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<VPUIP::arch40xx::DefaultHWOptions>(
            "default-hw-mode-vpuip", "VPUIP dialect part of Default HW pipeline",
            [](mlir::OpPassManager& pm, const VPUIP::arch40xx::DefaultHWOptions& options) {
                VPUIP::arch40xx::buildDefaultHWPipeline(pm, options);
            });
}
