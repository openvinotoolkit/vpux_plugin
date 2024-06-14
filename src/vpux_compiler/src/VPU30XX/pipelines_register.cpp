//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU30XX/pipelines_register.hpp"
#include "vpux/compiler/VPU30XX/conversion.hpp"
#include "vpux/compiler/VPU30XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/VPU30XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/VPU30XX/pipelines.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace vpux;

//
// PipelineRegistry30XX::registerPipelines
//

void PipelineRegistry30XX::registerPipelines() {
    mlir::PassPipelineRegistration<>("ShaveCodeGen", "Compile both from IE to VPUIP and from IERT to LLVM for NPU30XX",
                                     [](mlir::OpPassManager& pm) {
                                         buildShaveCodeGenPipeline30XX(pm);
                                     });

    mlir::PassPipelineRegistration<ReferenceSWOptions30XX>(
            "reference-sw-mode", "Compile IE Network in Reference Software mode (SW only execution) for NPU30XX",
            [](mlir::OpPassManager& pm, const ReferenceSWOptions30XX& options) {
                VPU::buildInitCompilerPipeline(pm,
                                               {VPU::ArchKind::NPU30XX, VPU::CompilationMode::ReferenceSW, options});

                buildReferenceSWModePipeline(pm, options);
            });

    mlir::PassPipelineRegistration<ReferenceHWOptions30XX>(
            "reference-hw-mode", "Compile IE Network in Reference Hardware mode (HW and SW execution) for NPU30XX",
            [](mlir::OpPassManager& pm, const ReferenceHWOptions30XX& options) {
                VPU::buildInitCompilerPipeline(pm,
                                               {VPU::ArchKind::NPU30XX, VPU::CompilationMode::ReferenceHW, options});

                buildReferenceHWModePipeline(pm, options);
            });

    mlir::PassPipelineRegistration<DefaultHWOptions30XX>(
            "default-hw-mode", "Compile IE Network in Default Hardware mode (HW and SW execution) for NPU30XX",
            [](mlir::OpPassManager& pm, const DefaultHWOptions30XX& options) {
                VPU::buildInitCompilerPipeline(pm, {VPU::ArchKind::NPU30XX, VPU::CompilationMode::DefaultHW, options});

                buildDefaultHWModePipeline(pm, options);
            });
    vpux::IE::arch30xx::registerIEPipelines();
    vpux::VPU::arch30xx::registerVPUPipelines();
    vpux::VPUIP::arch30xx::registerVPUIPPipelines();
    vpux::arch30xx::registerConversionPipeline();
}
