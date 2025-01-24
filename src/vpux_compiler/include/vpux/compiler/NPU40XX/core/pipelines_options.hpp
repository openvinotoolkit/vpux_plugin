//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/pipelines_options.hpp"

namespace vpux {
namespace arch40xx {

//
// DefaultHWOptionsDeviceBase (for all dialects in 40xx)
// This class must be inherited by all dialect-base options
// to avoid confusion when we have the same option for IE and the VPU dialect, but with a different value
//

struct DefaultHWOptionsDeviceBase : public virtual vpux::DefaultHWOptionsBase {
    StrOption enableActivationSparsity{*this, "enable-activation-sparsity",
                                       llvm::cl::desc("Enable activation sparsity"), llvm::cl::init("auto")};

    BoolOption enableWeightsSparsity{*this, "enable-weights-sparsity", llvm::cl::desc("Enable weights sparsity"),
                                     llvm::cl::init(true)};

    BoolOption enableSEPtrsOperations{*this, "enable-se-ptrs-operations",
                                      llvm::cl::desc("Enable storage element pointer operations"),
                                      llvm::cl::init(true)};

    BoolOption enableExperimentalSEPtrsOperations{*this, "enable-experimental-se-ptrs-operations",
                                                  llvm::cl::desc("Enable the experimental operation of SEP"),
                                                  llvm::cl::init(false)};

    BoolOption enableM2I{*this, "enable-m2i", llvm::cl::desc("Enable M2I passes"), llvm::cl::init(false)};

    BoolOption enableExplicitDistributionInfoAttr{
            *this, "enable-explicit-distributed-attr",
            llvm::cl::desc("Enable DistributionInfoAttr with explicit per cluster memory/compute shapes & offsets"),
            llvm::cl::init(true)};

    StrOption dpuDryRun{*this, "dpu-dry-run",
                        llvm::cl::desc("Patch DPU tasks to disable their functionality (none|stub|strip)"),
                        llvm::cl::init("none")};

    BoolOption shaveDryRun{*this, "shave-dry-run", llvm::cl::desc("Enable shave dry run stripping"),
                           llvm::cl::init(false)};

    BoolOption enablePartialWorkloadManagement{*this, "enable-partial-workload-management",
                                               llvm::cl::desc("Enable partial workload management"),
                                               llvm::cl::init(true)};

    IntOption wlmOptimizationThreshold{*this, "wlm-barriers-threshold",
                                       llvm::cl::desc("Threshold for WLM optimization"),
                                       llvm::cl::init(VIRTUAL_BARRIER_THRESHOLD_WLM)};

    mlir::detail::PassOptions::Option<WlmVpurtEnqueueMode> wlmVpurtEnqueue{
            *this, "wlm-vpurt-enqueue",
            ::llvm::cl::desc("Option for enabling WLM enqueue barriers search algorithm at VPURT. To be used only for "
                             "experiments."),
            ::llvm::cl::init(WlmVpurtEnqueueMode::DISABLED),
            ::llvm::cl::values(clEnumValN(WlmVpurtEnqueueMode::ENABLED, "ENABLED",
                                          "WLM enqueue barriers search algorithm at VPURT ENABLED"),
                               clEnumValN(WlmVpurtEnqueueMode::DISABLED, "DISABLED",
                                          "WLM enqueue barriers search algorithm at VPURT DISABLED. Use LCA based "
                                          "enqueue algorithm at VPUMI"))};

    BoolOption enableGroupedMatMul{*this, "enable-grouped-matmul",
                                   llvm::cl::desc("Enable execution of grouped MatMul as a single operation."),
                                   llvm::cl::init(true)};

    BoolOption enableOutputEnsurance{
            *this, "enable-output-ensurance",
            llvm::cl::desc(
                    "Enable output size ensurance when checking nce op shapes in EnsureNCEOpsSizeRequirements pass"),
            llvm::cl::init(true)};
};

//
// MCAndTilingOptionsDevice options
//

struct MCAndTilingOptionsDevice : public vpux::MCAndTilingOptionsBase {
    MCAndTilingOptionsDevice() {
        enableExplicitDistributionInfoAttr = true;
    }
};

}  // namespace arch40xx
}  // namespace vpux
