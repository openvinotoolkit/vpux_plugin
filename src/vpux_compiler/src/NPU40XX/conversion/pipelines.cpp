//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/conversion.hpp"

#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUASM/passes.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/utils.hpp"

#include <npu_40xx_nnrt.hpp>
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/abi_version.hpp"

#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// buildLowerVPUIP2ELFPipeline
//

void vpux::arch40xx::buildLowerVPUIP2ELFPipeline(mlir::OpPassManager& pm,
                                                 const BackendCompilationOptions40XX& backendCompilationOptions,
                                                 Logger log, VPU::DPUDryRunMode dpuDryRunMode) {
    log.info("BackendCompilationOptions:\n"
             "  enablePartialWorkloadManagement = {0}\n"
             "  wlmVpurtEnqueue = {1}\n"
             "  wlmOptimizationThreshold = {2}\n"
             "  enableMemorySideCache = {3}\n"
             "  enableDMAProfiling = {4}\n"
             "  enableShaveDDRAccessOptimization = {5}\n",
             backendCompilationOptions.enablePartialWorkloadManagement,
             backendCompilationOptions.wlmVpurtEnqueue == WlmVpurtEnqueueMode::ENABLED,
             backendCompilationOptions.wlmOptimizationThreshold, backendCompilationOptions.enableMemorySideCache,
             backendCompilationOptions.enableDMAProfiling, backendCompilationOptions.enableShaveDDRAccessOptimization);

    pm.addPass(VPUMI40XX::createAddPlatformInfoPass(log));
    pm.addPass(createConvertVPUIP2VPUMI40XXPass(log, backendCompilationOptions.enableMemorySideCache,
                                                backendCompilationOptions.allocateShaveStackFrames));
    auto dmaProfilingMode =
            getDMAProfilingMode(VPU::ArchKind::NPU40XX, backendCompilationOptions.enableDMAProfiling.getValue());
    pm.addPass(VPUMI40XX::createSetupProfilingVPUMI40XXPass(dmaProfilingMode, log));
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(ELF::createAddABIVersionPass(log, NPUReg40XX::ABI_VERSION_MAJOR, NPUReg40XX::ABI_VERSION_MINOR,
                                            NPUReg40XX::ABI_VERSION_PATCH));
    elfSubsetPipelineVPUMI(pm, backendCompilationOptions.enablePartialWorkloadManagement,
                           backendCompilationOptions.wlmVpurtEnqueue,
                           backendCompilationOptions.enableDumpStatisticsOfWlmOps, log);

    // To support forward compatibility between UD2024.44 and API version 11.4.10,
    // compiler by default set previous API version (11.4.10)
    pm.addPass(VPUMI40XX::createAddMappedInferenceVersionOpPass(log, VPUMI40XX::NNRT_API_UD2024_44_MAJOR_VERSION,
                                                                VPUMI40XX::NNRT_API_UD2024_44_MINOR_VERSION,
                                                                VPUMI40XX::NNRT_API_UD2024_44_PATCH_VERSION));

    // In case if IR contains logic which requires API version increasing, compiler will update the version and create
    // blob with correspongind NNRT API version value. In that case new blob will be backward compatible, but not
    // forward.
    pm.addPass(VPUMI40XX::createUpdateMappedInferenceVersionOpPass(log));

    elfSubsetPipelineVPUASM(pm, backendCompilationOptions.enablePartialWorkloadManagement, log);

    pm.addPass(VPUIPDPU::createExpandDPUConfigPass(log));
    pm.addPass(ELF::createUpdateELFSectionFlagsPass(log, backendCompilationOptions.enableShaveDDRAccessOptimization));
    pm.addPass(createConvertVPUASM2NPUReg40XXPass(log, backendCompilationOptions.enablePartialWorkloadManagement));
    pm.addPass(createConvertVPUIPDPU2NPUReg40XXPass(log, dpuDryRunMode));
    pm.addPass(ELF::createSetOpOffsetsPass(log, backendCompilationOptions.enablePartialWorkloadManagement));
    pm.addPass(ELF::createAddELFRelocationsPass(log));
    pm.addPass(ELF::createRemoveEmptyELFSectionsPass(log));
}

//
// buildElfSubsetPipelineVPUMI
//

void vpux::arch40xx::elfSubsetPipelineVPUMI(mlir::OpPassManager& pm, bool enablePartialWorkloadManagement,
                                            WlmVpurtEnqueueMode wlmVpurtEnqueue, bool enableDumpStatisticsOfWlmOps,
                                            const Logger& log) {
    if (!enablePartialWorkloadManagement) {
        pm.addPass(VPUMI40XX::createBarrierComputationPass(log));
        pm.addPass(VPUMI40XX::createLinkAllOpsPass(log));
        pm.addPass(VPUASM::createHoistInputOutputsPass(log));
        pm.addPass(VPUMI40XX::createResolveTaskLocationPass(log));
        pm.addPass(VPUMI40XX::reorderMappedInferenceOpsPass(log));
    } else {
        pm.addPass(VPUMI40XX::reorderMappedInferenceOpsPass(log));

        pm.addPass(VPUMI40XX::createBarrierTopologicalMappingPass(log));
        pm.addPass(VPUMI40XX::createGroupExecutionOpsPass(log));
        pm.addPass(VPUMI40XX::createAddFetchOpsPass(log));
        pm.addPass(VPUMI40XX::createResolveWLMTaskLocationPass(log));
        pm.addPass(VPUMI40XX::createUnGroupExecutionOpsPass(log));
        pm.addPass(VPUMI40XX::createPropagateFinalBarrierPass(log));
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(VPUMI40XX::createNextSameIdAssignmentPass(log));
        pm.addPass(VPUMI40XX::createAddEnqueueOpsPass(wlmVpurtEnqueue, log));
        pm.addPass(VPUMI40XX::createUnrollFetchTaskOpsPass(log));
        pm.addPass(VPUMI40XX::createAddBootstrapOpsPass(log));
        if (wlmVpurtEnqueue == WlmVpurtEnqueueMode::ENABLED) {
            pm.addPass(VPUMI40XX::createAddInitialBarrierConfigurationOps(log));
        }
        pm.addPass(VPUMI40XX::createSplitEnqueueOpsPass(log));
        pm.addPass(VPUMI40XX::createLinkEnqueueTargetsPass(log));
        pm.addPass(VPUMI40XX::createUnrollEnqueueOpsPass(log));
        pm.addPass(VPUMI40XX::reorderMappedInferenceOpsPass(log));

        if (enableDumpStatisticsOfWlmOps) {
            pm.addPass(VPUMI40XX::createDumpStatisticsOfWlmOpsPass(log));
        }

        pm.addPass(VPUASM::createHoistInputOutputsPass(log));
    }
}

//
// buildElfSubsetPipelineVPUASM
//

void vpux::arch40xx::elfSubsetPipelineVPUASM(mlir::OpPassManager& pm, bool enablePartialWorkloadManagement,
                                             const Logger& log) {
    pm.addPass(createConvertVPUMI40XX2VPUASMPass(log, enablePartialWorkloadManagement));
    pm.addPass(ELF::createAddInnerSectionPaddingPass(log));
    pm.addPass(ELF::createAddELFSymbolTablePass(log));
    pm.addPass(ELF::createSetEntryPointPass(log));
    pm.addPass(ELF::createAddNetworkMetadataPass(log));
    pm.addPass(VPUASM::createAddProfilingSectionPass(log));
}

//
// registerConversionPipelines40XX
//

void vpux::arch40xx::registerConversionPipeline() {
    mlir::PassPipelineRegistration<>("lower-IE-to-VPU", "Performs full lowering from the IE Dialect to VPU Dialect",
                                     [](mlir::OpPassManager& pm) {
                                         vpux::arch37xx::buildLowerIE2VPUPipeline(pm);
                                     });

    mlir::PassPipelineRegistration<vpux::DefaultHWOptions40XX>(
            "lower-VPU-to-VPUIP",
            "Performs full lowering from the VPU Dialect to VPUIP Dialect, SW operations are converted to SWKernelOp",
            [](mlir::OpPassManager& pm, const vpux::DefaultHWOptions40XX& options) {
                vpux::arch37xx::buildLowerVPU2VPUIPPipeline(pm, options.enableInPlaceBufferization);
            });

    mlir::PassPipelineRegistration<BackendCompilationOptions40XX>(
            "lower-VPUIP-to-ELF", "Performs full lowering from the VPUIP Dialect to the VPUMI40XX and ELF Dialects",
            [](mlir::OpPassManager& pm, const BackendCompilationOptions40XX& options) {
                vpux::arch40xx::buildLowerVPUIP2ELFPipeline(pm, options);
            });
}
