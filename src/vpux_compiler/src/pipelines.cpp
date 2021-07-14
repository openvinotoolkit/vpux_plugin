//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/pipelines.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IERT/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// Common utilities
//

namespace {

template <VPUIP::PhysicalMemory KIND>
mlir::Attribute getMemSpace(mlir::MLIRContext* ctx, StringRef) {
    return VPUIP::PhysicalMemoryAttr::get(ctx, KIND);
}

void buildIECommonPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(mlir::createCanonicalizerPass(getDefaultGreedyRewriteConfig()));
    IE::buildAdjustForVPUPipeline(pm, log);
    pm.addPass(IE::createUseUserPrecisionPass(log));
    pm.addPass(IE::createUseUserLayout(log));
    pm.addPass(IE::createAdjustLayoutsPass(log));
    pm.addPass(IE::createOptimizeReordersPass(log));
    pm.addPass(mlir::createCanonicalizerPass(getDefaultGreedyRewriteConfig()));
}

void buildIEReferenceLowPrecisionPipeline(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(IE::createSplitFakeQuantPass(log));
    pm.addPass(IE::createDequantizeConstPass(log));
    pm.addPass(IE::createMergeFakeQuantPass(log));
}

void buildIERTAllocationPipelineForDDR(mlir::OpPassManager& pm, Logger log) {
    pm.addPass(createDeallocPlacementPass(log));
    pm.addPass(IERT::createSetInternalMemorySpacePass(getMemSpace<VPUIP::PhysicalMemory::DDR>, log));
    pm.addPass(IERT::createStaticAllocationPass(getMemSpace<VPUIP::PhysicalMemory::DDR>, log));
}

}  // namespace

//
// ReferenceMode
//

void vpux::buildReferenceModePipeline(mlir::OpPassManager& pm, Logger log) {
    // IE Dialect level
    buildIECommonPipeline(pm, log);
    buildIEReferenceLowPrecisionPipeline(pm, log);

    // Lower IE->IERT
    buildLowerIE2IERTPipeline(pm, log);

    // IERT Dialect level
    pm.addPass(createComposeSubViewPass(log));
    buildIERTAllocationPipelineForDDR(pm, log);
    IERT::buildAsyncSchedulingPipeline(pm, log);

    // Lower IERT->VPUIP (SW mode)
    buildLowerIERT2VPUIPPipeline(pm, log);

    // VPUIP Dialect level
    pm.addPass(VPUIP::createAssignPhysicalBarriersPass(log));
}

//
// HardwareMode
//

void vpux::buildHardwareModePipeline(mlir::OpPassManager& pm, Logger log) {
    // IE Dialect level
    IE::buildHWOpsConversionPipeline(pm, log);
    buildIECommonPipeline(pm, log);
    IE::buildLowPrecisionPipeline(pm, log);

    // Lower IE->IERT
    buildLowerIE2IERTPipeline(pm, log);

    // Partially lower IERT->VPUIP (NCE Operations only)
    pm.addPass(IERT::createCMXTilingPass(log));
    pm.addPass(mlir::createCanonicalizerPass(getDefaultGreedyRewriteConfig()));
    pm.addPass(createConvertToNCEOpsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(getDefaultGreedyRewriteConfig()));

    // IERT Dialect level (cont.)
    pm.addPass(createComposeSubViewPass(log));
    buildIERTAllocationPipelineForDDR(pm, log);
    pm.addPass(IERT::createStaticAllocationPass(getMemSpace<VPUIP::PhysicalMemory::CMX_NN>, log));
    pm.addPass(VPUIP::createConvertWeightsTableOp2ConstPass(log));
    IERT::buildAsyncSchedulingPipeline(pm, log);

    // Finally lower remaining IERT->VPUIP (SW mode)
    buildLowerIERT2VPUIPPipeline(pm, log);

    // VPUIP Dialect level
    pm.addPass(VPUIP::createAssignPhysicalBarriersPass(log));
}

//
// registerPipelines
//

void vpux::registerPipelines() {
    mlir::PassPipelineRegistration<>("reference-mode", "Compile IE Network in Reference mode (SW only execution)",
                                     [](mlir::OpPassManager& pm) {
                                         buildReferenceModePipeline(pm);
                                     });

    mlir::PassPipelineRegistration<>("hardware-mode", "Compile IE Network in Hardware mode (HW and SW execution)",
                                     [](mlir::OpPassManager& pm) {
                                         buildHardwareModePipeline(pm);
                                     });
}
