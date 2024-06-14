//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU37XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/core/pipelines_options.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

namespace vpux {
namespace VPU {
namespace arch40xx {

//
// Passes
//

std::unique_ptr<mlir::Pass> createFuseM2IOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertM2IOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createComputeNCEInputWorkloadsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMoveConvertAroundViewLikeOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCorrectNCEWorkloadsPass(Logger log = Logger::global());

void buildIncrementalPipeline(mlir::OpPassManager& pm, const vpux::MCAndTilingOptionsBase& options,
                              Logger log = Logger::global());

//
// DefaultHWOptions
//

struct DefaultHWOptions : public VPU::DefaultHWOptionsDialectBase, virtual vpux::arch40xx::DefaultHWOptionsDeviceBase {
    StrOption actSparsityProfile{*this, "act-sparsity-profile", llvm::cl::desc("Activation sparsity profile"),
                                 llvm::cl::init("S0")};

    BoolOption enableVPUNNCost{*this, "vpunn-cost",
                               llvm::cl::desc("Use VPUNN cost model to get the best tiling strategy"),
                               llvm::cl::init(true)};

    BoolOption enableOutputPipelining{*this, "output-pipelining", llvm::cl::desc("Enable output pipelining"),
                                      llvm::cl::init(true)};

    BoolOption enablePartialWorkloadManagement{*this, "enable-partial-workload-management",
                                               llvm::cl::desc("Enable partial workload management"),
                                               llvm::cl::init(false)};

    IntOption wlmOptimizationThreshold{*this, "wlm-barriers-threshold",
                                       llvm::cl::desc("Threshold for WLM optimization"), llvm::cl::init(3000)};
};

void buildDefaultHWPipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options, Logger log = Logger::global());

//
// registerVPUPipelines
//

void registerVPUPipelines();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/NPU40XX/dialect/VPU/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/NPU40XX/dialect/VPU/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace arch40xx
}  // namespace VPU
}  // namespace vpux
