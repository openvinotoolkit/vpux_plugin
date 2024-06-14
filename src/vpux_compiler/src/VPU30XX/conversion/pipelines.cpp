//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU30XX/conversion.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// LowerIE2VPU
//

void vpux::arch30xx::buildLowerIE2VPUPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(vpux::arch30xx::createConvertIEToVPUNCEPass(log));
    pm.addPass(createConvertLayers2VPUPass(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// LowerVPU2VPUIPUPA
//

void vpux::arch30xx::buildLowerVPU2VPUIPPipeline(mlir::OpPassManager& pm, Logger log) {
    const auto grc = getDefaultGreedyRewriteConfig();

    pm.addPass(createConvertSWLayers2VPUIPUPAPass(log));
    pm.addPass(createOneShotBufferizeVPU2VPUIPPass());
    pm.addPass(createAddBuffersForNetResults(log));
    pm.addPass(mlir::createCanonicalizerPass(grc));
}

//
// registerConversionPipelines
//

void vpux::arch30xx::registerConversionPipeline() {
    mlir::PassPipelineRegistration<>("lower-IE-to-VPU", "Performs full lowering from the IE Dialect to VPU Dialect",
                                     [](mlir::OpPassManager& pm) {
                                         vpux::arch30xx::buildLowerIE2VPUPipeline(pm);
                                     });

    mlir::PassPipelineRegistration<>(
            "lower-VPU-to-VPUIP",
            "Performs full lowering from the VPU Dialect to VPUIP Dialect, SW operations are converted to UPAOp",
            [](mlir::OpPassManager& pm) {
                vpux::arch30xx::buildLowerVPU2VPUIPPipeline(pm);
            });
}
