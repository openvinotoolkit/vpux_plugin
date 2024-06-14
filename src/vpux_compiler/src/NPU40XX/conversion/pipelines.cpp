//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/conversion.hpp"

#include "vpux/compiler/NPU40XX/dialect/ELF/passes.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/passes.hpp"
#include "vpux/compiler/dialect/VPUASM/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// buildLowerVPUIP2ELFPipeline
//

void vpux::arch40xx::buildLowerVPUIP2ELFPipeline(mlir::OpPassManager& pm,
                                                 const BackendCompilationOptions40XX& backendCompilationOptions,
                                                 Logger log, VPU::DPUDryRunMode dpuDryRunMode) {
    pm.addPass(createConvertVPUIP2VPUMI40XXPass(log, backendCompilationOptions.enableMemorySideCache));
    pm.addPass(VPUMI40XX::createSetupProfilingVPUMI40XXPass(log));
    pm.addPass(mlir::createCanonicalizerPass());

    elfSubsetPipeline(pm, backendCompilationOptions, log);

    pm.addPass(ELF::createMoveOpsIntoSectionsPass(log));
    pm.addPass(ELF::createAddInnerSectionPaddingPass(log));
    pm.addPass(ELF::createAddELFSymbolTablePass(log));
    pm.addPass(createMoveIOBuffersToSectionsPass(log));
    pm.addPass(ELF::createSetEntryPointPass(log));
    pm.addPass(ELF::createAddNetworkMetadataPass(log));
    pm.addPass(VPUASM::createAddProfilingSectionPass(log));
    pm.addPass(VPUIPDPU::createExpandDPUConfigPass(log));

    pm.addPass(
            createConvertVPUASM2NPUReg40XXRelocsPass(log, backendCompilationOptions.enablePartialWorkloadManagement));
    pm.addPass(createConvertVPUIPDPU2NPUReg40XXPass(log, dpuDryRunMode));
    pm.addPass(ELF::createSetOpOffsetsPass(log));
    pm.addPass(ELF::createAddELFRelocationsPass(log));
    pm.addPass(ELF::createUpdateELFSectionFlagsPass(log));
    pm.addPass(createConvertVPUASM2NPUReg40XXPass(log));
    pm.addPass(ELF::createRemoveEmptyELFSectionsPass(log));
}

//
// buildElfSubsetPipeline
//

void vpux::arch40xx::elfSubsetPipeline(mlir::OpPassManager& pm,
                                       const BackendCompilationOptions40XX& backendCompilationOptions, Logger log) {
    if (!backendCompilationOptions.enablePartialWorkloadManagement) {
        pm.addPass(VPUMI40XX::createBarrierComputationPass(log));
        pm.addPass(VPUMI40XX::createLinkAllOpsPass(log));
        pm.addPass(VPUASM::createHoistInputOutputsPass(log));
        pm.addPass(VPUMI40XX::createResolveTaskLocationPass(log));
        pm.addPass(VPUMI40XX::reorderMappedInferenceOpsPass(log));
        pm.addPass(createConvertVPUMI40XX2VPUASMPass(log));
    } else {
        pm.addPass(VPUMI40XX::reorderMappedInferenceOpsPass(log));

        pm.addPass(VPUMI40XX::createBarrierTopologicalMappingPass(backendCompilationOptions.wlmOptimizationThreshold,
                                                                  log));
        pm.addPass(VPUMI40XX::createGroupExecutionOpsPass(log));
        pm.addPass(VPUMI40XX::createWorkloadManagementPass(log));
        pm.addPass(VPUMI40XX::createResolveWLMTaskLocationPass(log));
        pm.addPass(VPUMI40XX::createUnGroupExecutionOpsPass(log));
        pm.addPass(VPUMI40XX::createPropagateFinalBarrierPass(log));

        pm.addPass(mlir::createCanonicalizerPass());

        pm.addPass(VPUMI40XX::createNextSameIdAssignmentPass(log));
        pm.addPass(VPUMI40XX::createAddEnqueueOpsPass(log));
        pm.addPass(VPUMI40XX::createUnrollFetchTaskOpsPass(log));
        pm.addPass(VPUMI40XX::createAddBootstrapOpsPass(log));
        pm.addPass(VPUMI40XX::createLinkEnqueueTargetsPass(log));
        pm.addPass(VPUMI40XX::createUnrollEnqueueOpsPass(log));

        pm.addPass(VPUMI40XX::reorderMappedInferenceOpsPass(log));

        pm.addPass(VPUASM::createHoistInputOutputsPass(log));
        pm.addPass(createConvertVPUMI40XX2VPUASMPass(log));

        pm.addPass(VPUASM::createBarriersToManagedBarriersPass(log));
    }
}

//
// registerConversionPipelines40XX
//

void vpux::arch40xx::registerConversionPipeline() {
    mlir::PassPipelineRegistration<>("lower-IE-to-VPU", "Performs full lowering from the IE Dialect to VPU Dialect",
                                     [](mlir::OpPassManager& pm) {
                                         vpux::arch37xx::buildLowerIE2VPUPipeline(pm);
                                     });

    mlir::PassPipelineRegistration<>(
            "lower-VPU-to-VPUIP",
            "Performs full lowering from the VPU Dialect to VPUIP Dialect, SW operations are converted to SWKernelOp",
            [](mlir::OpPassManager& pm) {
                vpux::arch37xx::buildLowerVPU2VPUIPPipeline(pm);
            });

    mlir::PassPipelineRegistration<BackendCompilationOptions40XX>(
            "lower-VPUIP-to-ELF", "Performs full lowering from the VPUIP Dialect to the VPUMI40XX and ELF Dialects",
            [](mlir::OpPassManager& pm, const BackendCompilationOptions40XX& options) {
                vpux::arch40xx::buildLowerVPUIP2ELFPipeline(pm, options);
            });
}
