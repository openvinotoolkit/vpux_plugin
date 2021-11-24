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

#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// AdjustPrecision
//

void vpux::IE::buildAdjustPrecisionPipeline(mlir::OpPassManager& pm, const AdjustPrecisionOptions& options,
                                            Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createConvertPrecisionToFP16Pass(log));
    pm.addPass(IE::createConvertPrecisionToI32Pass(log));
    if (options.enableUseUserPrecision) {
        pm.addPass(IE::createUseUserPrecisionPass(log));
    }
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// AdjustLayout
//

void vpux::IE::buildAdjustLayoutPipeline(mlir::OpPassManager& pm, const AdjustLayoutOptions& options, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    if (options.enableUseUserLayout) {
        pm.addPass(IE::createUseUserLayout(log));
    }
    pm.addPass(IE::createAdjustLayoutsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    if (options.enableOptimizeReorders) {
        pm.addPass(IE::createOptimizeReordersPass(log));
    }
    pm.addPass(IE::createConvertToMemPermutePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// AdjustForVPU
//

void vpux::IE::buildAdjustForVPUPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createConvertTile2PerAxisTilePass(log));
    pm.addPass(IE::createConvertShapeTo4DPass(log));
    pm.addPass(IE::createConvertConv1DToConv2DPass(log));
    pm.addPass(IE::createConvertPaddingsToFloorModePass(log));
    pm.addPass(IE::createResolveStridedSlicePass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));

    pm.addPass(IE::createFusePostOpsPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// LowPrecision
//

void vpux::IE::buildLowPrecisionPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(IE::createSplitFakeQuantPass(log));
    pm.addPass(IE::createFuseConvertWithQuantizePass(log));
    pm.addPass(IE::createFuseQuantizedOpsPass(log));
    pm.addPass(IE::createConvertWeightsToU8Pass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
    pm.addPass(IE::createDequantizeConstPass(log));
    pm.addPass(IE::createConvertQuantizeOpsToEltwisePass(log));
    pm.addPass(IE::createMergeFakeQuantPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// registerIEPipelines
//

void vpux::IE::registerIEPipelines() {
    mlir::PassPipelineRegistration<AdjustPrecisionOptions>(
            "adjust-precision", "Adjust IR precision for VPU target",
            [](mlir::OpPassManager& pm, const AdjustPrecisionOptions& options) {
                IE::buildAdjustPrecisionPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<AdjustLayoutOptions>(
            "adjust-layout", "Adjust IR layout for VPU target",
            [](mlir::OpPassManager& pm, const AdjustLayoutOptions& options) {
                IE::buildAdjustLayoutPipeline(pm, options);
            });

    mlir::PassPipelineRegistration<>("adjust-for-vpu", "Adjust IE Dialect IR for VPU target",
                                     [](mlir::OpPassManager& pm) {
                                         IE::buildAdjustForVPUPipeline(pm);
                                     });

    mlir::PassPipelineRegistration<>("low-precision", "Low precision transformations", [](mlir::OpPassManager& pm) {
        IE::buildLowPrecisionPipeline(pm);
    });
}
