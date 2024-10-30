//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/pipelines.hpp"

#include "vpux/compiler/NPU37XX/conversion.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/profiling/common.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// ReferenceSWMode
//

void vpux::buildReferenceSWModePipeline(mlir::OpPassManager& pm, const ReferenceSWOptions37XX& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Level 3 : Topology
    IE::arch37xx::buildInitialLowPrecisionTransformationsPipeline(pm, IE::LowPrecisionTransformOptions(options), log);
    IE::arch37xx::buildInitialTransformationsPipeline(pm, IE::arch37xx::TransformOptions(options), log);
    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    // Resolve group quant MatMul pattern
    pm.addPass(IE::createUniquifyOpsPass(log));
    pm.addPass(IE::createMergeParallelFullyConnectedPass(log));
    pm.addPass(IE::createUnrollGroupQuantizePass(log));
    pm.addPass(IE::createUnrollFullyConnectedPass(log));
    if (options.fuseScalesToAccumulate) {
        pm.addPass(IE::createFuseScalesToAccumulatePass(log));
    }
    pm.addPass(IE::createConvertMatMulToConvPass(log));
    if (options.enableConvertFCToConv) {
        pm.addPass(IE::createConvertFCToConvPass(log));
    }

    pm.addPass(IE::createResolveStridedSlicePass(log));
    pm.addPass(IE::createConvertNceOpsTo4DPass(log));
    pm.addPass(IE::createConvertShapeTo4DPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(
            IE::createConvertToSpatialOpPass(false, isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
    pm.addPass(IE::createConvertGRNToNormalizeL2Pass(log));
    pm.addPass(IE::createResolveScatterUpdateByTransposePass(log));
    IE::buildAdjustForVPUPipeline(pm, IE::AdjustForVPUOptions(options), log);

    pm.addPass(IE::createSplitFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createDequantizeConstPass(log));
    if (options.enableMergeFakeQuant) {
        pm.addPass(IE::createMergeFakeQuantPass(log));
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::arch37xx::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);
    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));

    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    // Lowering to VPU
    pm.addPass(createConvertLayers2VPUPass(log));
    pm.addPass(VPU::createDetectionOutputDecompositionPass(log));
    pm.addPass(VPU::arch37xx::createSplitRealDFTOpsPass(log));
    pm.addPass(VPU::createSplitGRUSequencePass(log));
    pm.addPass(VPU::arch37xx::createDecomposeMVNPass(log));
    pm.addPass(VPU::arch37xx::createAddProposalAuxiliaryBufferPass(log));

    pm.addPass(VPU::createTilingStrategyAssignmentPass(/*enablePrefetchTiling=*/false, false, "true", log));
    pm.addPass(VPU::arch37xx::createApplyTilingMVN1SumPass(log));
    pm.addPass(VPU::createApplyTilingPass(log));
    pm.addPass(VPU::createLegalizeDynamicShapeConcatForSWLayersPass(log));

    // Lowering to VPUIP
    vpux::arch37xx::buildLowerVPU2VPUIPPipeline(pm, log);

    // Level 2 : Abstract RunTime

    pm.addPass(VPUIP::createSetMemorySpacePass(VPU::getMemKind<VPU::MemoryKind::DDR>, log));

    pm.addPass(VPUIP::createCopyOpTilingPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableProfiling && options.enableSWProfiling) {
        pm.addPass(VPUIP::createActShaveProfilingPass(VPU::getMemKind<VPU::MemoryKind::CMX_NN>, log));
    }

    pm.addPass(VPUIP::createUngroupBoundedBuffersPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPUIP::createConvertTransferOpsToDMAsPass(log));

    VPUIP::buildAsyncSchedulingPipeline(pm, log);

    if (options.enableSWKernelPrefetchingReserveMem) {
        pm.addPass(VPUIP::createSWKernelPrefetchingReserveMemPass(log));
    }

    pm.addPass(VPUIP::createAddCopyBetweenSWKernelsAndNetworkIOPass(log));

    pm.addPass(VPUIP::createStaticAllocationPass(VPU::getMemKind<VPU::MemoryKind::CMX_NN>, log));
    pm.addPass(VPUIP::createStaticAllocationPass(VPU::getMemKind<VPU::MemoryKind::DDR>, log));
    pm.addPass(VPUIP::createCollectUsedMemoryPass());
    pm.addPass(VPUIP::createLinearizationPass(log));
    pm.addPass(VPUIP::createOptimizeAsyncDepsPass(log));

    pm.addPass(VPUIP::arch37xx::createAddSwKernelCacheHandlingOpsPass(log));

    VPUIP::buildHardwareAdaptationPipeline(pm, log);

    // Level 1 : VPU RunTime

    if (options.enableProfiling) {
        pm.addPass(VPUIP::createCaptureWorkpointPass(log));
        pm.addPass(VPUIP::createGroupProfilingBuffersPass(log));
        pm.addPass(createMoveDeclarationsToTopPass(log));
    }

    pm.addPass(VPURT::arch37xx::createAddUpdateBarrierForSwKernelsPass(log));
    pm.addPass(VPURT::createAssignPhysicalBarriersPass(false, options.enableColorBinPhysicalBarrierAssignment,
                                                       std::nullopt, log));
    pm.addPass(VPURT::createBarrierSimulationPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.nest<mlir::func::FuncOp>().addNestedPass<Const::DeclareOp>(Const::createConstantFoldingPass());
    pm.addPass(VPUIP::createDumpStatisticsOfTaskOpsPass(log));
}

//
// ReferenceHWMode
//

void vpux::buildReferenceHWModePipeline(mlir::OpPassManager& pm, const ReferenceHWOptions37XX& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    // Level 3 : Topology
    if (options.logOpOptimizations) {
        pm.addPass(IE::createLogOpOptimizationsPass());
    }
    IE::arch37xx::buildInitialLowPrecisionTransformationsPipeline(pm, IE::LowPrecisionTransformOptions(options), log);
    IE::arch37xx::buildInitialTransformationsPipeline(pm, IE::arch37xx::TransformOptions(options), log);
    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    IE::buildOperationConversionPipeline(pm, IE::OperationConversionOptions(options), log);

    pm.addPass(IE::createConvertNceOpsTo4DPass(log));
    if (options.enableHandleLargeKernel) {
        pm.addPass(IE::createAdjustMaxPoolInputShapePass(log));
        pm.addPass(IE::createHandleLargeKernelsPass(log));
    }
    pm.addPass(IE::createHandleExcludePadForAvgPoolPass(log));
    if (options.enableConvertAvgPoolToDWConv) {
        pm.addPass(IE::createConvertAvgPoolToDWConvPass(log));
    }

    pm.addPass(IE::createAdaptShapesForScaleShiftPass(log));
    pm.addPass(IE::createResolveStridedSlicePass(log));
    pm.addPass(IE::createConvertShapeTo4DPass(log));
    pm.addPass(IE::createSwapTransposeConcatPass(log));
    pm.addPass(IE::createConvertSplitConcatToTransposePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(
            IE::createConvertToSpatialOpPass(false, isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
    pm.addPass(IE::createSwapOperationsPass(isOptionEnabled(options.enableSEPtrsOperations) ||
                                                    isOptionEnabled(options.enableExperimentalSEPtrsOperations),
                                            log));
    pm.addPass(IE::createSwapPadLayerPass(log));
    pm.addPass(IE::createConvertSubtractToAddPass(log));
    pm.addPass(IE::createConvertToScaleShiftPass(log));
    pm.addPass(IE::createBroadcastInputForAddPass(log));
    pm.addPass(IE::createConvertGRNToNormalizeL2Pass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    // E#79878: Solve eltwise single layer test failure.
    // SwapOperations pass may generate non-4D AddOp.
    // If AddOp appears here means that it cannot be fused into NCE task.
    // So convert it's shape to 4D and then convert this AddOp to ScaleShift.
    pm.addPass(IE::createConvertShapeTo4DPass(log));
    pm.addPass(IE::createConvertToScaleShiftPass(log));
    pm.addPass(IE::createResolveScatterUpdateByTransposePass(log));
    pm.addPass(IE::createConvertGroupConvToConvPass(log));
    pm.addPass(IE::createSwapOperationsPass(isOptionEnabled(options.enableSEPtrsOperations) ||
                                                    isOptionEnabled(options.enableExperimentalSEPtrsOperations),
                                            log));

    IE::buildAdjustForVPUPipeline(pm, IE::AdjustForVPUOptions(options), log);

    if (options.enableSwapTransposeWithFQ) {
        pm.addPass(IE::createSwapTransposeWithFQPass(log));
    }
    if (options.enableSplitConvWithMultipleFQ) {
        pm.addPass(IE::createSplitConvWithMultipleFQPass(log));
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableHandleLargeKernel) {
        pm.addPass(IE::createHandleLargeKernelsPass(log));
    }
    if (options.enableHandleLargeStrides) {
        pm.addPass(IE::createHandleLargeStridesPass(log));
    }
    if (options.enableHandleAsymmetricStrides) {
        pm.addPass(IE::createHandleAsymmetricStridesPass(log));
    }
    if (options.enableHandleLargePads) {
        pm.addPass(IE::createHandleLargePadsPass(log));
    }
    pm.addPass(IE::createConvertGroupConvToConvPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    if (options.enableOptimizeScaleShiftToDWConv) {
        IE::buildScaleShiftProcessingPipeline(pm, log);
    }

    pm.addPass(IE::createFuseActivationOpsPass(options.enableFuseClampOperations, log));
    pm.addPass(IE::createConvertStridedSlice2ConvPass(log));
    if (options.enableLowPrecision) {
        IE::arch37xx::buildLowPrecisionPipeline(pm, IE::LowPrecisionOptions(options), log);
        pm.addPass(IE::createConvertShapeTo4DPass(log));
        pm.addPass(IE::createSwapViewOpAndClampPass(log));
    }
    IE::arch37xx::buildOptimizeActivationsPipeline(pm, IE::OptimizeActivationsOptions(options), log);

    if (options.enableSEPtrsOperations && options.enableSplitBilinerIntoHAndW) {
        pm.addPass(IE::createSplitBilinerIntoHAndWPass(log));
    }

    if (options.enableBilinearInterpolateOnDPU) {
        pm.addPass(IE::arch37xx::createMapBilinearInterpolateOnDPUPass(isOptionEnabled(options.enableSEPtrsOperations),
                                                                       log));
    }

    pm.addPass(IE::createConvertBatchedLayerTo1NPass(log));
    pm.addPass(IE::arch37xx::createUnrollBatchPass(log));

    if (options.enableUpstreamSlice) {
        pm.addPass(IE::createUpstreamSlicePass(log));
    }

    pm.addPass(IE::createSwapMVNWithTransposePass(log));

    IE::arch37xx::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);
    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));

    if (options.enableFusePermuteQuantize) {
        pm.addPass(IE::createFusePermuteQuantizePass(false, log));
        pm.addPass(IE::createConvertReorderToPermuteQuantizePass(log));
    }

    if (options.enableExpandActivationChannels) {
        pm.addPass(IE::createAdjustGroupConvShapePass(log));
    }
    IE::arch37xx::buildMemPermutePositioningPipeline(pm, IE::MemPermutePositioningOptions(options), log);

    if (options.enableExpandActivationChannels) {
        if (options.enableAdjustConvShapePass) {
            pm.addPass(IE::createOptimizeAvgPoolWithUnalignedChannelsPass(log));
            pm.addPass(IE::createAdjustConvolutionShapePass(log));
        }
        pm.addPass(IE::arch37xx::createExpandActivationChannelsPass(
                /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
                /*seExperimentalOpsEnabled=*/isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
        pm.addPass(mlir::createCanonicalizerPass(grc));

        if (options.enableOptimizeSliceExpand) {
            pm.addPass(IE::arch37xx::createOptimizeSliceExpandPass(log));
        }

        pm.addPass(IE::createAdjustConvolutionWeightsPass(log));
        pm.addPass(IE::createAdjustConvolutionInputShapePass(log));
        pm.addPass(IE::createAdjustInputShapePass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
        if (options.enableOptimizeSliceExpand) {
            pm.addPass(IE::arch37xx::createOptimizeSliceExpandPass(log));
        }

        if (options.enableOptimizeReorders) {
            pm.addPass(IE::createOptimizeReordersPass(
                    /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
                    /*seExperimentalOpsEnabled=*/isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
            pm.addPass(IE::createOptimizeReordersAcrossFunctionCallsPass(
                    /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
                    /*seExperimentalOpsEnabled=*/isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
            pm.addPass(IE::createUniquifyOpsPass(log));
            pm.addPass(IE::createPropagateAffineReshapePass(log));
            pm.addPass(IE::createUniquifyBranchesPass(log));
        }

        if (options.enableFusePermuteQuantizeExpand) {
            pm.addPass(IE::arch37xx::createPropagateExpandPass(log));
            pm.addPass(IE::arch37xx::createFusePermuteQuantizeExpandPass(log));
        }
    }

    pm.addPass(IE::createSwapOperationsPass(isOptionEnabled(options.enableSEPtrsOperations) ||
                                                    isOptionEnabled(options.enableExperimentalSEPtrsOperations),
                                            log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createConvertSplitConcatToTransposePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    IE::arch37xx::buildMemPermuteProcessingPipeline(pm, log);
    pm.addPass(IE::createRemoveViewLikeOpsChainPass(log));
    pm.addPass(IE::createOptimizeOpSlicePass(log));
    pm.addPass(IE::createUniquifyOpsPass(log));

    if (options.enableExpandActivationChannels) {
        pm.addPass(IE::createExpandActivationWidthPass(log));
        pm.addPass(IE::createAdjustInputShapePass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
        if (options.enableOptimizeSliceExpand) {
            pm.addPass(IE::arch37xx::createOptimizeSliceExpandPass(log));
        }
        pm.addPass(IE::createPropagateAffineReshapePass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
    }
    if (options.enableOptimizeSliceWithStride) {
        pm.addPass(IE::createOptimizeSliceWithStridePass(log));
        if (options.enableAdjustConvShapePass) {
            pm.addPass(IE::createAdjustConvolutionShapePass(log));
        }
    }
    if (options.enableConvertExpandToConvPass) {
        pm.addPass(IE::createConvertExpandToConvPass(log));
    }
    pm.addPass(IE::createOptimizeIdentityPoolPass(log));
    if (options.logOpOptimizations) {
        pm.addPass(IE::createLogOpOptimizationsPass());
    }

    // Lowering to VPU
    vpux::arch37xx::buildLowerIE2VPUPipeline(pm, log);
    pm.addPass(VPU::arch37xx::createAdjustForOptimizedLayersPass(log));

    pm.addPass(VPU::createDetectionOutputDecompositionPass(log));
    pm.addPass(VPU::arch37xx::createSplitRealDFTOpsPass(log));
    pm.addPass(VPU::arch37xx::createAddProposalAuxiliaryBufferPass(log));

    if (options.enableSEPtrsOperations || options.enableExperimentalSEPtrsOperations) {
        pm.addPass(VPU::createSplitSEOpsPass(
                /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
                /*seExperimentalOpsEnabled=*/isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
        pm.addPass(VPU::createLowerOpsToSENCEPass(
                /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
                /*seExperimentalOpsEnabled=*/isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
    }

    pm.addPass(VPU::createFuseClampPass(log));

    pm.addPass(VPU::createEnsureNCEOpsSizeRequirementsPass(log));
    pm.addPass(VPU::createOptimizeConcatPass(log));

    if (options.enableWeightsSparsity) {
        VPU::buildWeightsSparsityPipeline(pm, VPU::WeightsSparsityOptions(options), log);
    }
    pm.addPass(VPU::createAddExplicitPaddingBeforeNCEPermutePass(log));
    if (VPU::isActSparsityEnabled(options.enableActivationSparsity)) {
        VPU::buildActivationSparsityPipeline(pm, VPU::ActivationSparsityOptions(options), log);
        pm.addPass(VPU::createLowerSparsityOpsPass(/*fakeSparsify=*/false, log));
    }

    if (options.enableInPlaceEltwise) {
        pm.addPass(VPU::createDetectInPlaceEltwisePass(log));
    }

    if (options.enableSMPipeline) {
        VPU::buildSMPipeline(pm, vpux::MCAndTilingOptionsBase(options), log);
    } else {
        VPU::arch37xx::buildIncrementalPipeline(pm, vpux::MCAndTilingOptionsBase(options), log);
    }

    pm.addPass(VPU::createOptimizeSharedInputCopyForConcatPass(log));
    pm.addPass(VPU::createOptimizeConcatPass(log));
    pm.addPass(VPU::createAdjustMemorySpacePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(VPU::createWrapDistributedOpsInNCEClusterTiling(log));

    pm.addPass(VPU::createCMXConcatPass(log, options.supportNCEOpInsertion));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPU::createSplitNCEOpsOntoWorkloadsPass(log));
    pm.addPass(VPU::arch37xx::createCorrectNCEWorkloadsPass(log));
    pm.addPass(VPU::createResolveEltwiseWithZTiledWorkloadsPass(log));

    // Lowering to VPUIP
    vpux::arch37xx::buildLowerVPU2VPUIPPipeline(pm, log);
    pm.addPass(VPUIP::createTileActShaveKernelTaskPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    if (options.enableOptimizeCopies || options.enableOpsAsDMA) {
        // This pass is a part of "copy optimization pipeline", but need to be done before because
        // WrapWithPermuteAsNNDMA depends on it.
        pm.addPass(VPUIP::createMovePureViewOpBeforeCopyPass(log));
    }
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
    }

    VPUIP::arch37xx::buildOptimizeCopiesPipeline(pm, VPUIP::arch37xx::OptimizeCopiesOptions(options), log);

    pm.addPass(VPUIP::createInsertCopyForEltwiseInPlaceInputPass(log));
    if (options.enableOpsAsDMA) {
        pm.addPass(VPUIP::createConvertToDMAPass(log));
    }
    pm.addPass(VPUIP::createCopyOpTilingPass(log));

    pm.addPass(mlir::createCanonicalizerPass(grc));
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

    if (options.enableProfiling) {
        auto dmaProfilingMode = getDMAProfilingMode(VPU::ArchKind::NPU37XX, options.enableDMAProfiling.getValue());
        pm.addPass(VPUIP::createDMATaskProfilingReserveMemPass(dmaProfilingMode, log));
    }

    if (options.enableSWKernelPrefetchingReserveMem) {
        pm.addPass(VPUIP::createSWKernelPrefetchingReserveMemPass(log));
    }

    pm.addPass(VPUIP::createAddCopyBetweenSWKernelsAndNetworkIOPass(log));

    pm.addPass(VPUIP::createCalculateAsyncRegionCycleCostPass(log));

    VPUIP::arch37xx::buildMemoryAllocationPipeline(pm, VPUIP::arch37xx::MemoryAllocationOptions(options), log);

    pm.addPass(VPUIP::createOptimizeAsyncDepsPass(log));

    if (options.enablePopulateWeightTableWithShave) {
        pm.addPass(VPUIP::createPatchPopulateWeightTableWithShavePass(log));
    }

    // Handle WeightsTable, which requires statically allocated memory
    pm.addPass(VPUIP::createPatchWeightsTablePass(log));

    pm.addPass(VPUIP::arch37xx::createAddSwKernelCacheHandlingOpsPass(log));

    VPUIP::buildHardwareAdaptationPipeline(pm, log);

    // Level 1 : VPU RunTime
    pm.addPass(VPUIP::createUnrollSwKernelPass(log));

    pm.addPass(VPUIP::arch37xx::createUnrollClusterTilingPass(log));
    pm.addPass(VPUIP::createNNDMATilingPass(log));
    if (options.enableWeightsSparsity) {
        pm.addPass(VPUIP::createFlattenSparseWeightsTypesPass(log));
    }
    if (VPU::isActSparsityEnabled(options.enableActivationSparsity) || options.enableSEPtrsOperations) {
        pm.addPass(VPUIP::createComputeSESizesPass(/*onlyInputsConcatOverC=*/
                                                   false, log));
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

    VPURT::buildBarrierLegalizationPipeline(pm, /* wlmFlag */ false, std::nullopt, /* unevenVariantSplitFlag */ false,
                                            log);

    pm.addPass(VPURT::arch37xx::createAddUpdateBarrierForSwKernelsPass(log));

    pm.addPass(Const::createApplySwizzlingPass());

    pm.addPass(VPUIP::createResolveDMAWithSwizzlingPass(log));

    if (options.enableCompressWeightsBTC) {
        pm.addPass(VPUIP::createCompressWeightsBTCPass(log));
    }

    if (options.enableProfiling) {
        auto dmaProfilingMode = getDMAProfilingMode(VPU::ArchKind::NPU37XX, options.enableDMAProfiling.getValue());
        pm.addPass(VPUIP::createDMATaskProfilingAfterBarrierSchedPass(dmaProfilingMode, log));
        pm.addPass(VPUIP::createCaptureWorkpointPass(log));
        pm.addPass(VPUIP::createGroupProfilingBuffersPass(log));
        pm.addPass(createMoveDeclarationsToTopPass(log));
    }

    pm.addPass(VPURT::createAssignPhysicalBarriersPass(false, options.enableColorBinPhysicalBarrierAssignment,
                                                       std::nullopt, log));
    pm.addPass(VPURT::createBarrierSimulationPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.nest<mlir::func::FuncOp>().addNestedPass<Const::DeclareOp>(Const::createConstantFoldingPass());

    if (options.enableIntermediateBufferOutput) {
        pm.addPass(VPURT::createIntermediateBufferOutputPass(log));
    }

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

//
// ShaveCodeGen
//

void vpux::buildShaveCodeGenPipeline37XX(mlir::OpPassManager& pm, Logger log) {
    log.trace("Entered buildShaveCodeGenPipeline37XX()");

    // Code copied from the buildDefaultHWModePipeline().
    const auto grc = getDefaultGreedyRewriteConfig();

    // Level 3 : Topology

    const DefaultHWOptions37XX options;  // TODO: takeout (normally)
    IE::arch37xx::buildInitialLowPrecisionTransformationsPipeline(pm, IE::LowPrecisionTransformOptions(options), log);
    IE::arch37xx::buildInitialTransformationsPipeline(pm, IE::arch37xx::TransformOptions(options), log);
    IE::buildAdjustPrecisionPipeline(pm, IE::AdjustPrecisionOptions(options), log);

    pm.addPass(IE::createConvertAssignReadValueToReturnsAndInputs(log));

    IE::buildOperationConversionPipeline(pm, IE::OperationConversionOptions(options), log);

    pm.addPass(IE::createConvertNceOpsTo4DPass(log));
    if (options.enableHandleLargeKernel) {
        pm.addPass(IE::createAdjustMaxPoolInputShapePass(log));
        pm.addPass(IE::createHandleLargeKernelsPass(log));
    }
    pm.addPass(IE::createHandleExcludePadForAvgPoolPass(log));
    if (options.enableConvertAvgPoolToDWConv) {
        pm.addPass(IE::createConvertAvgPoolToDWConvPass(log));
    }

    pm.addPass(IE::createAdaptShapesForScaleShiftPass(log));
    pm.addPass(IE::createResolveStridedSlicePass(log));
    pm.addPass(IE::createConvertShapeTo4DPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(
            IE::createConvertToSpatialOpPass(false, isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
    pm.addPass(IE::createSwapOperationsPass(isOptionEnabled(options.enableSEPtrsOperations) ||
                                                    isOptionEnabled(options.enableExperimentalSEPtrsOperations),
                                            log));
    pm.addPass(IE::createSwapPadLayerPass(log));
    pm.addPass(IE::createConvertSubtractToAddPass(log));
    pm.addPass(IE::createConvertToScaleShiftPass(log));
    pm.addPass(IE::createBroadcastInputForAddPass(log));
    pm.addPass(IE::createConvertGRNToNormalizeL2Pass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createResolveScatterUpdateByTransposePass(log));
    pm.addPass(IE::createConvertGroupConvToConvPass(log));
    pm.addPass(IE::createSwapOperationsPass(isOptionEnabled(options.enableSEPtrsOperations) ||
                                                    isOptionEnabled(options.enableExperimentalSEPtrsOperations),
                                            log));

    IE::buildAdjustForVPUPipeline(pm, IE::AdjustForVPUOptions(options), log);

    if (options.enableSwapTransposeWithFQ) {
        pm.addPass(IE::createSwapTransposeWithFQPass(log));
    }
    if (options.enableSplitConvWithMultipleFQ) {
        pm.addPass(IE::createSplitConvWithMultipleFQPass(log));
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));

    if (options.enableHandleLargeKernel) {
        pm.addPass(IE::createHandleLargeKernelsPass(log));
    }
    if (options.enableHandleLargeStrides) {
        pm.addPass(IE::createHandleLargeStridesPass(log));
    }
    if (options.enableHandleAsymmetricStrides) {
        pm.addPass(IE::createHandleAsymmetricStridesPass(log));
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));
    if (options.enableOptimizeScaleShiftToDWConv) {
        IE::buildScaleShiftProcessingPipeline(pm, log);
    }

    pm.addPass(IE::createFuseActivationOpsPass(options.enableFuseClampOperations, log));
    pm.addPass(IE::createConvertStridedSlice2ConvPass(log));
    if (options.enableLowPrecision) {
        IE::arch37xx::buildLowPrecisionPipeline(pm, IE::LowPrecisionOptions(options), log);
        pm.addPass(IE::createConvertShapeTo4DPass(log));
        pm.addPass(IE::createSwapViewOpAndClampPass(log));
    }
    pm.addPass(IE::createFuseActivationOpsPass(options.enableFuseClampOperations, log));
    pm.addPass(IE::createConvertBatchedLayerTo1NPass(log));
    pm.addPass(IE::arch37xx::createUnrollBatchPass(log));

    if (options.enableUpstreamSlice) {
        pm.addPass(IE::createUpstreamSlicePass(log));
    }

    pm.addPass(IE::createSwapMVNWithTransposePass(log));

    IE::arch37xx::buildAdjustLayoutPipeline(pm, IE::AdjustLayoutOptions(options), log);

    if (options.enableFusePermuteQuantize) {
        pm.addPass(IE::createFusePermuteQuantizePass(false, log));
    }

    if (options.enableExpandActivationChannels) {
        pm.addPass(IE::createAdjustGroupConvShapePass(log));
        if (options.enableAdjustConvShapePass) {
            pm.addPass(IE::createOptimizeAvgPoolWithUnalignedChannelsPass(log));
            pm.addPass(IE::createAdjustConvolutionShapePass(log));
        }
        pm.addPass(IE::arch37xx::createExpandActivationChannelsPass(
                /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
                /*seExperimentalOpsEnabled=*/isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
        pm.addPass(mlir::createCanonicalizerPass(grc));

        if (options.enableOptimizeSliceExpand) {
            pm.addPass(IE::arch37xx::createOptimizeSliceExpandPass(log));
        }

        pm.addPass(IE::createAdjustConvolutionWeightsPass(log));
        pm.addPass(IE::createAdjustInputShapePass(log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
        if (options.enableOptimizeSliceExpand) {
            pm.addPass(IE::arch37xx::createOptimizeSliceExpandPass(log));
        }

        if (options.enableOptimizeReorders) {
            pm.addPass(IE::createOptimizeReordersPass(
                    /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
                    /*seExperimentalOpsEnabled=*/isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
            pm.addPass(IE::createOptimizeReordersAcrossFunctionCallsPass(
                    /*seOpsEnabled=*/isOptionEnabled(options.enableSEPtrsOperations),
                    /*seExperimentalOpsEnabled=*/isOptionEnabled(options.enableExperimentalSEPtrsOperations), log));
            pm.addPass(IE::createUniquifyOpsPass(log));
            pm.addPass(IE::createPropagateAffineReshapePass(log));
            pm.addPass(IE::createUniquifyBranchesPass(log));
        }

        if (options.enableFusePermuteQuantizeExpand) {
            pm.addPass(IE::arch37xx::createPropagateExpandPass(log));
            pm.addPass(IE::arch37xx::createFusePermuteQuantizeExpandPass(log));
        }
    }

    pm.addPass(IE::createSwapOperationsPass(isOptionEnabled(options.enableSEPtrsOperations) ||
                                                    isOptionEnabled(options.enableExperimentalSEPtrsOperations),
                                            log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createMovePermutePostEltwisePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createLegalizeNDMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    // Here it ends the code added from buildDefaultHWModePipeline().

    // Here we add new code
    buildLowerIE2IERTPipeline(pm, log);

    pm.addPass(vpux::createConvertSWLayers2AffinePass(log));
    pm.addPass(vpux::createConvertAffine2LLVMPass(log));

    // TODO: lowering to shave ASM
    // TODO: need pass IERT to VPUIP for e.g. function @main

    log.trace("Exiting buildShaveCodeGenPipeline37XX()");
}

//
// DefaultHWMode
//

void vpux::buildDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions37XX& options, Logger log) {
    IE::arch37xx::buildDefaultHWPipeline(pm, options, log);

    // Lowering to VPU
    vpux::arch37xx::buildLowerIE2VPUPipeline(pm, log);
    VPU::arch37xx::buildDefaultHWPipeline(pm, options, log);

    // Lowering to VPUIP
    vpux::arch37xx::buildLowerVPU2VPUIPPipeline(pm, log);
    VPUIP::arch37xx::buildDefaultHWPipeline(pm, options, log);
}
