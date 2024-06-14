//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "intel_npu/al/config/common.hpp"
#include "intel_npu/al/config/compiler.hpp"

#include "vpux/compiler/VPU30XX/pipeline_strategy.hpp"
#include "vpux/compiler/VPU30XX/pipelines.hpp"

#include "vpux/compiler/options_mapper.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

using namespace vpux;

//
// PipelineStrategy30XX::buildPipeline
//

void PipelineStrategy30XX::buildPipeline(mlir::PassManager& pm, const Config& config, mlir::TimingScope& rootTiming,
                                         Logger log) {
    auto buildTiming = rootTiming.nest("Build compilation pipeline");

    const auto initCompilerOptions = getInitCompilerOptions(config);
    VPU::buildInitCompilerPipeline(pm, initCompilerOptions, log.nest());

    const auto enableProfiling = config.get<intel_npu::PERF_COUNT>();
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        const auto options = ReferenceSWOptions30XX::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "buildPipeline failed to parse COMPILATION_MODE_PARAMS");
        options->enableProfiling = enableProfiling;
        buildReferenceSWModePipeline(pm, *options, log.nest());
    } else if (compilationMode == VPU::CompilationMode::ReferenceHW) {
        const auto options = ReferenceHWOptions30XX::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "buildPipeline failed to parse COMPILATION_MODE_PARAMS");
        options->enableProfiling = enableProfiling;
        buildReferenceHWModePipeline(pm, *options, log.nest());
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        const auto options = DefaultHWOptions30XX::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "buildPipeline failed to parse COMPILATION_MODE_PARAMS");
        options->enableProfiling = enableProfiling;

        buildDefaultHWModePipeline(pm, *options, log.nest());
    } else if (compilationMode == VPU::CompilationMode::ShaveCodeGen) {
        buildShaveCodeGenPipeline30XX(pm, log.nest());
    } else {
        VPUX_THROW("Unsupported compilation mode '{0}'", compilationMode);
    }
}

void PipelineStrategy30XX::buildELFPipeline(mlir::PassManager&, const Config&, mlir::TimingScope&, Logger) {
}
