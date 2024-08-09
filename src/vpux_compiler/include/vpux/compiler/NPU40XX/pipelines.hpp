//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/core/pipelines_options.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

namespace vpux {

//
// ReferenceSWOptions40XX
//

struct ReferenceSWOptions40XX final : public ReferenceSWOptions<ReferenceSWOptions40XX> {
    BoolOption enableConvertFFTToConv{*this, "convert-fft-to-conv", llvm::cl::desc("Enable convert-fft-to-conv pass"),
                                      llvm::cl::init(false)};

    BoolOption enableStartBarrier{*this, "enable-start-barrier", llvm::cl::desc("Enable start barrier"),
                                  llvm::cl::init(true)};

    BoolOption enableFinalBarrier{*this, "enable-final-barrier", llvm::cl::desc("Enable final barrier"),
                                  llvm::cl::init(true)};
};

void buildReferenceSWModePipeline(mlir::OpPassManager& pm, const ReferenceSWOptions40XX& options,
                                  Logger log = Logger::global());

//
// ReferenceHWOptions40XX
//

struct ReferenceHWOptions40XX final : public ReferenceHWOptions<ReferenceHWOptions40XX> {
    BoolOption enableConvertFFTToConv{*this, "convert-fft-to-conv", llvm::cl::desc("Enable convert-fft-to-conv pass"),
                                      llvm::cl::init(true)};

    BoolOption enableFusePermuteQuantize{*this, "fuse-permute-quantize",
                                         llvm::cl::desc("Enable fuse-permute-quantize pass"), llvm::cl::init(true)};

    BoolOption enableM2I{*this, "enable-m2i", llvm::cl::desc("Enable M2I passes"), llvm::cl::init(false)};

    BoolOption enableFusePermuteQuantizeExpand{*this, "fuse-permute-quantize-expand",
                                               llvm::cl::desc("Enable fuse-permute-quantize-expand pass"),
                                               llvm::cl::init(true)};

    BoolOption enableWeightsSwizzling{*this, "enable-weights-swizzling", ::llvm::cl::desc("Enable weights swizzling"),
                                      ::llvm::cl::init(false)};

    BoolOption enableActivationSwizzling{*this, "enable-activation-swizzling",
                                         ::llvm::cl::desc("Enable activation swizzling"), ::llvm::cl::init(false)};

    StrOption actSparsityProfile{*this, "act-sparsity-profile", llvm::cl::desc("Activation sparsity profile"),
                                 llvm::cl::init("S0")};

    StrOption enableActivationSparsity{*this, "enable-activation-sparsity",
                                       llvm::cl::desc("Enable activation sparsity"), llvm::cl::init("false")};

    BoolOption enableWeightsSparsity{*this, "enable-weights-sparsity", llvm::cl::desc("Enable weights sparsity"),
                                     llvm::cl::init(false)};

    BoolOption enableSEPtrsOperations{*this, "enable-se-ptrs-operations",
                                      llvm::cl::desc("Enable storage element pointer operations"),
                                      llvm::cl::init(false)};

    BoolOption enableExperimentalSEPtrsOperations{*this, "enable-experimental-se-ptrs-operations",
                                                  llvm::cl::desc("Enable the experimental operation of SEP"),
                                                  llvm::cl::init(false)};

    BoolOption enableCompressActivationSpill{*this, "compress-activation-spill",
                                             ::llvm::cl::desc("Enable compress-activation-spill feature"),
                                             ::llvm::cl::init(false)};

    BoolOption enableFuseClampOperations{*this, "enable-fuse-clamp-op", llvm::cl::desc("Enable fuse clamp operations"),
                                         llvm::cl::init(false)};

    BoolOption enableVPUNNCost{*this, "vpunn-cost",
                               llvm::cl::desc("Use VPUNN cost model to get the best tiling strategy"),
                               llvm::cl::init(false)};

    BoolOption enableOutputPipelining{*this, "output-pipelining", llvm::cl::desc("Enable output pipelining"),
                                      llvm::cl::init(false)};

    BoolOption enableExplicitDistributedTensorAttr{
            *this, "enable-explicit-distributed-attr",
            llvm::cl::desc("Enable DistributedTensorAttr with explicit per cluster memory/compute shapes & offsets"),
            llvm::cl::init(true)};

    BoolOption enableGroupedMatMul{*this, "enable-grouped-matmul",
                                   llvm::cl::desc("Enable execution of grouped MatMul as a single operation."),
                                   llvm::cl::init(false)};

    BoolOption supportNCEOpInsertion{
            *this, "support-nce-op-insertion",
            llvm::cl::desc("Insert a new NCE operation with single user for CMX-Concat to handle the"
                           "complex case when parent NCE has an extra non-Copy user."),
            llvm::cl::init(true)};

    BoolOption enableStartBarrier{*this, "enable-start-barrier", llvm::cl::desc("Enable start barrier"),
                                  llvm::cl::init(true)};

    BoolOption enableFinalBarrier{*this, "enable-final-barrier", llvm::cl::desc("Enable final barrier"),
                                  llvm::cl::init(true)};
};

void buildReferenceHWModePipeline(mlir::OpPassManager& pm, const ReferenceHWOptions40XX& options,
                                  Logger log = Logger::global());

//
// DefaultHWOptions40XX
//

struct DefaultHWOptions40XX final :
        public IE::arch40xx::DefaultHWOptions,
        VPU::arch40xx::DefaultHWOptions,
        VPUIP::arch40xx::DefaultHWOptions,
        mlir::PassPipelineOptions<DefaultHWOptions40XX> {
    // Due to multiple inheritance, 'DefaultHWOptions40XX' has multiple definitions of 'createFromString' method
    // here we assume that we are interested in a "final" method that includes parameters from all parent classes
    using mlir::PassPipelineOptions<DefaultHWOptions40XX>::createFromString;
};

//
// BackendCompilationOptions40XX
//

struct BackendCompilationOptions40XX final : public mlir::PassPipelineOptions<BackendCompilationOptions40XX> {
    BoolOption enableMemorySideCache{*this, "enable-memory-side-cache", llvm::cl::desc("Enable memory side cache"),
                                     llvm::cl::init(false)};
    BoolOption enablePartialWorkloadManagement{*this, "enable-partial-workload-management",
                                               llvm::cl::desc("Enable partial workload management"),
                                               llvm::cl::init(true)};
    StrOption enableDMAProfiling{*this, "dma-profiling",
                                 llvm::cl::desc("Enable DMA task profiling (true|static|false)"),
                                 llvm::cl::init("false")};

    IntOption wlmOptimizationThreshold{*this, "wlm-barriers-threshold",
                                       llvm::cl::desc("Threshold for WLM optimization"), llvm::cl::init(3000)};
};

void buildShaveCodeGenPipeline40XX(mlir::OpPassManager& pm, Logger log = Logger::global());

void buildDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions40XX& options,
                                Logger log = Logger::global());
}  // namespace vpux
