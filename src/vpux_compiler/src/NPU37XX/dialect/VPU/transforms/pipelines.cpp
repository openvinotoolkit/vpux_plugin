//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

void vpux::VPU::arch37xx::buildIncrementalPipeline(mlir::OpPassManager& pm, const vpux::MCAndTilingOptionsBase& options,
                                                   Logger log) {
    pm.addPass(VPU::arch37xx::createDecomposeMVNPass(log));

    pm.addPass(VPU::createMultiClusterStrategyAssignmentPass(options.enablePrefetching, options.enableMCSideLoadDump,
                                                             options.opTilingCacheThreshold, options.modelHash, log));

    pm.addPass(VPU::createManualStrategyUtilsPass(options.writeStrategyToJson, writeStrategyFileLocation,
                                                  options.readStrategyFromJson, readStrategyFileLocation,
                                                  options.enableMCSideLoadDump, options.modelHash, log));

    pm.addPass(VPU::createSplitGRUSequencePass(log));
    pm.addPass(VPU::arch37xx::createApplyTilingMVN1SumPass(options.enablePrefetching, log));

    VPU::buildTilingPipeline(pm, VPU::TilingOptions(options), log);
    pm.addPass(VPU::createMakeOpsWithDistributedTensorPass(options.enableExplicitDistributionInfoAttr, log));

    pm.addPass(VPU::createComputeInterpolateCoordinatesPass(options.enableExplicitDistributionInfoAttr, log));
    pm.addPass(VPU::createRemoveOutputSparseToAvoidSuboptimalDPUWorkloadsPass(log));

    pm.addPass(VPU::createMakeDistributedCopiesPass(log));
    pm.addPass(VPU::createAdjustDistributedTensorAroundOpsPass(log));
}

//
// DefaultHWPipeline
//

void vpux::VPU::arch37xx::buildDefaultHWPipeline(mlir::OpPassManager& pm,
                                                 const VPU::arch37xx::DefaultHWOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    if (options.enableConcatRepeatingBlockOutlining) {
        pm.addPass(VPU::createConcatRepeatingBlocksOutliningPass(options.concatRepeatingBlockOutliningSeqLength, log));
        pm.addPass(mlir::createCanonicalizerPass(grc));
    }

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

    pm.addPass(VPU::createEnsureNCEOpsSizeRequirementsPass(true, log));
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

    pm.addPass(VPU::createCMXConcatPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(VPU::createSplitNCEOpsOntoWorkloadsPass(log));
    pm.addPass(VPU::createWrapDistributedOpsInNCEClusterTiling(log));
    pm.addPass(VPU::arch37xx::createCorrectNCEWorkloadsPass(log));
    pm.addPass(VPU::createResolveEltwiseWithZTiledWorkloadsPass(log));
    pm.addPass(VPU::createOutlineEntireMainContentPass(log));
}

void vpux::VPU::arch37xx::registerVPUPipelines() {
    mlir::PassPipelineRegistration<VPU::arch37xx::DefaultHWOptions>(
            "default-hw-mode-vpu", "VPU dialect part of Default HW pipeline",
            [](mlir::OpPassManager& pm, const VPU::arch37xx::DefaultHWOptions& options) {
                VPU::arch37xx::buildDefaultHWPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<vpux::arch37xx::MCAndTilingOptionsDevice>(
            "incremental-pipeline", "Apply Incremental Pipeline",
            [](mlir::OpPassManager& pm, const vpux::arch37xx::MCAndTilingOptionsDevice& options) {
                VPU::arch37xx::buildIncrementalPipeline(pm, vpux::MCAndTilingOptionsBase(options));
            });

    mlir::PassPipelineRegistration<vpux::arch37xx::MCAndTilingOptionsDevice>(
            "sm-pipeline", "Apply SM Pipeline",
            [](mlir::OpPassManager& pm, const vpux::arch37xx::MCAndTilingOptionsDevice& options) {
                VPU::buildSMPipeline(pm, vpux::MCAndTilingOptionsBase(options));
            });
}
