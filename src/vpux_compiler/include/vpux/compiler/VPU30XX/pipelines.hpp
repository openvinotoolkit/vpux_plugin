//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/VPU30XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/VPU30XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/VPU30XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/core/pipelines_options.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

namespace vpux {

//
// ReferenceSWOptions30XX
//

struct ReferenceSWOptions30XX final : public ReferenceSWOptions<ReferenceSWOptions30XX> {};

void buildReferenceSWModePipeline(mlir::OpPassManager& pm, const ReferenceSWOptions30XX& options,
                                  Logger log = Logger::global());

//
// ReferenceHWOptions30XX
//

struct ReferenceHWOptions30XX final : public ReferenceHWOptions<ReferenceHWOptions30XX> {
    BoolOption enableWeightsSparsity{*this, "enable-weights-sparsity", llvm::cl::desc("Enable weights sparsity"),
                                     llvm::cl::init(false)};

    BoolOption enableSEPtrsOperations{*this, "enable-se-ptrs-operations",
                                      llvm::cl::desc("Enable storage element pointer operations"),
                                      llvm::cl::init(false)};

    BoolOption enableExperimentalSEPtrsOperations{*this, "enable-experimental-se-ptrs-operations",
                                                  llvm::cl::desc("Enable the experimental operation of SEP"),
                                                  llvm::cl::init(false)};

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
            llvm::cl::init(false)};
};

void buildReferenceHWModePipeline(mlir::OpPassManager& pm, const ReferenceHWOptions30XX& options,
                                  Logger log = Logger::global());

//
// DefaultHWOptions30XX
//

struct DefaultHWOptions30XX final :
        public IE::arch30xx::DefaultHWOptions,
        VPU::arch30xx::DefaultHWOptions,
        VPUIP::arch30xx::DefaultHWOptions,
        mlir::PassPipelineOptions<DefaultHWOptions30XX> {
    // Due to multiple inheritance, 'DefaultHWOptions30XX' has multiple definitions of 'createFromString' method
    // here we assume that we are interested in a "final" method that includes parameters from all parent classes
    using mlir::PassPipelineOptions<DefaultHWOptions30XX>::createFromString;
};

void buildShaveCodeGenPipeline30XX(mlir::OpPassManager& pm, Logger log = Logger::global());

void buildDefaultHWModePipeline(mlir::OpPassManager& pm, const DefaultHWOptions30XX& options,
                                Logger log = Logger::global());

}  // namespace vpux
