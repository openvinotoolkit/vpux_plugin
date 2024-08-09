//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/Pass/PassManager.h>

namespace vpux {
namespace VPUIP {

struct OptimizeCopiesOptionsBase : mlir::PassPipelineOptions<OptimizeCopiesOptionsBase> {
    OptimizeCopiesOptionsBase() = default;

    BoolOption enableOptimizeCopies{*this, "optimize-copies", llvm::cl::desc("Enable optimize-copies pass"),
                                    llvm::cl::init(true)};
    BoolOption enableOptimizeConstCopies{*this, "optimize-const-copies", llvm::cl::desc("Enable optimize-const-copies"),
                                         llvm::cl::init(true)};
    BoolOption enableOpsAsDMA{*this, "enable-ops-as-dma",
                              llvm::cl::desc("Force using DMA transformations instead of SW ops"),
                              llvm::cl::init(true)};

    template <class OtherOptions>
    explicit OptimizeCopiesOptionsBase(const OtherOptions& options) {
        enableOptimizeCopies = options.enableOptimizeCopies;
        enableOptimizeConstCopies = options.enableOptimizeConstCopies;
        enableOpsAsDMA = options.enableOpsAsDMA;
    }
};

struct MemoryAllocationOptionsBase : mlir::PassPipelineOptions<MemoryAllocationOptionsBase> {
    BoolOption linearizeSchedule{*this, "linearize-schedule", llvm::cl::desc("Linearize tasks on all engines"),
                                 llvm::cl::init(false)};

    BoolOption enablePrefetching{*this, "prefetching",
                                 llvm::cl::desc("Enable prefetch tiling pass and prefetch scheduling"),
                                 llvm::cl::init(true)};

    BoolOption enablePipelining{*this, "pipelining",
                                llvm::cl::desc("Enable vertical fusion pipelining pass and schedule pipelining"),
                                llvm::cl::init(true)};

    BoolOption optimizeFragmentation{*this, "optimize-fragmentation",
                                     ::llvm::cl::desc("Enables compiler to optimize CMX fragmentation"),
                                     ::llvm::cl::init(true)};

    BoolOption optimizeDynamicSpilling{*this, "optimize-dynamic-spilling",
                                       ::llvm::cl::desc("Enables compiler to optimize dynamic spilling DMAs"),
                                       ::llvm::cl::init(true)};

    BoolOption enableGroupAsyncExecuteOps{*this, "group-async-execute-ops",
                                          llvm::cl::desc("Enable group-async-execute-ops pass"), llvm::cl::init(false)};

    MemoryAllocationOptionsBase() = default;

    template <class OtherOptions>
    explicit MemoryAllocationOptionsBase(const OtherOptions& options) {
        linearizeSchedule = options.linearizeSchedule;
        enablePrefetching = options.enablePrefetching;
        enablePipelining = options.enablePipelining;
        optimizeDynamicSpilling = options.optimizeDynamicSpilling;
        enableGroupAsyncExecuteOps = options.enableGroupAsyncExecuteOps;
    }
};

}  // namespace VPUIP
}  // namespace vpux
