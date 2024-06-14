//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/compiler.hpp"

#include "intel_npu/al/config/common.hpp"
#include "intel_npu/al/config/compiler.hpp"

#include "vpux/compiler/NPU40XX/conversion.hpp"
#include "vpux/compiler/NPU40XX/pipeline_strategy.hpp"
#include "vpux/compiler/NPU40XX/pipelines.hpp"

#include "vpux/compiler/options_mapper.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/dry_run_utils.hpp"

using namespace vpux;

//
// PipelineStrategy40XX::buildPipeline
//

void PipelineStrategy40XX::buildPipeline(mlir::PassManager& pm, const Config& config, mlir::TimingScope& rootTiming,
                                         Logger log) {
    auto buildTiming = rootTiming.nest("Build compilation pipeline");

    const auto initCompilerOptions = getInitCompilerOptions(config);
    const auto& numOfDPUGroups = initCompilerOptions.numberOfDPUGroups;
    const auto& numOfDMAPorts = initCompilerOptions.numberOfDMAPorts;

    VPUX_THROW_WHEN(
            numOfDPUGroups.hasValue() && numOfDMAPorts.hasValue() &&
                    numOfDMAPorts.getValue() > numOfDPUGroups.getValue(),
            "Requested configuration not supported by runtime. Number of DMA ports ({0}) larger than NCE clusters "
            "({1})",
            numOfDMAPorts.getValue(), numOfDPUGroups.getValue());

    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, log.nest());

    const auto enableProfiling = config.get<intel_npu::PERF_COUNT>();
    const auto compilationMode = getCompilationMode(config);
    const auto isElf = isELFEnabled(config);

    VPUX_THROW_WHEN(!isElf && numOfDMAPorts.hasValue() && numOfDMAPorts.getValue() > 1,
                    "With Graphfile backend only single DMA is supported");

    const auto backendCompilationOptions =
            BackendCompilationOptions40XX::createFromString(config.get<intel_npu::BACKEND_COMPILATION_PARAMS>());
    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        const auto options = ReferenceSWOptions40XX::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "buildPipeline failed to parse COMPILATION_MODE_PARAMS");
        options->enableProfiling = enableProfiling;
        buildReferenceSWModePipeline(pm, *options, log.nest());
    } else if (compilationMode == VPU::CompilationMode::ReferenceHW) {
        const auto options = ReferenceHWOptions40XX::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "buildPipeline failed to parse COMPILATION_MODE_PARAMS");
        options->enableProfiling = enableProfiling;
        buildReferenceHWModePipeline(pm, *options, log.nest());
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        const auto options = DefaultHWOptions40XX::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "buildPipeline failed to parse COMPILATION_MODE_PARAMS");
        options->enableProfiling = enableProfiling;
        options->enableConvertAvgPoolToDWConv = false;
        options->enableHandleAsymmetricStrides = false;
        options->enablePartialWorkloadManagement = backendCompilationOptions->enablePartialWorkloadManagement;

        // TODO: E-108844 Support Compressed activation with Partial workload management
        if (!isElf || backendCompilationOptions->enablePartialWorkloadManagement) {
            options->enableCompressActivationSpill = false;
        }
        buildDefaultHWModePipeline(pm, *options, log.nest());
    } else if (compilationMode == VPU::CompilationMode::ShaveCodeGen) {
        buildShaveCodeGenPipeline40XX(pm, log.nest());
    } else {
        VPUX_THROW("Unsupported compilation mode '{0}'", compilationMode);
    }
}

void PipelineStrategy40XX::buildELFPipeline(mlir::PassManager& pm, const Config& config, mlir::TimingScope& rootTiming,
                                            Logger log) {
    auto buildTiming = rootTiming.nest("Build compilation pipeline");

    auto dpuDryRunMode = VPU::DPUDryRunMode::NONE;
    const auto compilationMode = getCompilationMode(config);
    const auto backendCompilationOptions =
            BackendCompilationOptions40XX::createFromString(config.get<intel_npu::BACKEND_COMPILATION_PARAMS>());

    if (compilationMode == VPU::CompilationMode::DefaultHW) {
        const auto options = DefaultHWOptions40XX::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "build ELF pipeline failed to parse COMPILATION_MODE_PARAMS");
        dpuDryRunMode = VPU::getDPUDryRunMode(options->dpuDryRun);
    }
    arch40xx::buildLowerVPUIP2ELFPipeline(pm, *backendCompilationOptions, log.nest(), dpuDryRunMode);
}
