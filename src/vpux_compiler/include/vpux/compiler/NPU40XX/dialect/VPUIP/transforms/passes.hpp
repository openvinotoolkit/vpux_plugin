//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/core/pipelines_options.hpp"

#include "vpux/compiler/dialect/VPU/utils/dry_run_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/pipelines_options.hpp"

namespace vpux {
namespace VPUIP {
namespace arch40xx {

//
// Passes
//

std::unique_ptr<mlir::Pass> createComputeTaskStrippingPass(
        Logger log = Logger::global(), VPU::DPUDryRunMode dryRunStripTarget = VPU::DPUDryRunMode::NONE,
        bool shaveDryRun = false);

std::unique_ptr<mlir::Pass> createComputeHaloRegionForDPUTaskOpPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDMATaskProfilingHwDdrPass(
        DMAProfilingMode dmaProfilingMode = DMAProfilingMode::DISABLED, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConstantDpuProfHwpBasePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCompressSpillDmaPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDMAOutOfOrderOptimizationPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createUnrollClusterTilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeConvertDMAOpPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddStartBarrierPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDetectDMASplitCandidatePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitDMAToBalanceLoadPass(Logger log = Logger::global());

//
// Memory allocation pipeline
//

struct MemoryAllocationOptions final : public VPUIP::MemoryAllocationOptionsBase {
    BoolOption enableCompressActivationSpill{*this, "compress-activation-spill",
                                             ::llvm::cl::desc("Enable compress-activation-spill feature"),
                                             ::llvm::cl::init(false)};

    MemoryAllocationOptions() = default;

    template <class OtherOptions>
    MemoryAllocationOptions(const OtherOptions& options) {
        enableCompressActivationSpill = options.enableCompressActivationSpill;
    }
};

void buildMemoryAllocationPipeline(mlir::OpPassManager& pm, const MemoryAllocationOptions& options,
                                   Logger log = Logger::global());

//
// DefaultHWOptions
//

struct DefaultHWOptions :
        public VPUIP::DefaultHWOptionsDialectBase,
        virtual vpux::arch40xx::DefaultHWOptionsDeviceBase {
    // Enable for 40XX once RT will be ready, follow up #E95864
    StrOption enableDMAProfiling{*this, "dma-profiling",
                                 llvm::cl::desc("Enable DMA task profiling (true, false, static)"),
                                 ::llvm::cl::init("false")};

    BoolOption enableCompressWeightsBTC{*this, "compress-weights-btc", ::llvm::cl::desc("Enable compress-weights pass"),
                                        ::llvm::cl::init(false)};

    BoolOption enableWeightsSwizzling{*this, "enable-weights-swizzling", ::llvm::cl::desc("Enable weights swizzling"),
                                      ::llvm::cl::init(true)};

    BoolOption enableActivationSwizzling{*this, "enable-activation-swizzling",
                                         ::llvm::cl::desc("Enable activation swizzling"), ::llvm::cl::init(true)};

    BoolOption enableCompressActivationSpill{*this, "compress-activation-spill",
                                             ::llvm::cl::desc("Enable compress-activation-spill feature"),
                                             ::llvm::cl::init(false)};

    // TODO: E#118871 Switch this option to true by default
    BoolOption enableBarrierSchedWithFunctionOutlining{
            *this, "barrier-sched-with-function-outlining",
            llvm::cl::desc("Enable barrier scheduling passes with IR split into multiple functions"),
            llvm::cl::init(false)};

    BoolOption enableSWKernelPrefetchingReserveMem{
            *this, "enable-sw-kernel-prefetching-reserve-mem",
            ::llvm::cl::desc("Reserve memory at the end of CMX for SW Kernel data prefetching"),
            ::llvm::cl::init(true)};
};

void buildDefaultHWPipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options, Logger log = Logger::global());

//
// registerVPUIPPipelines
//

void registerVPUIPPipelines();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/NPU40XX/dialect/VPUIP/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/NPU40XX/dialect/VPUIP/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace arch40xx
}  // namespace VPUIP
}  // namespace vpux
