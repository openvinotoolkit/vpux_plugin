//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/pipelines_options.hpp"
#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/optional.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace vpux {
namespace VPUIP {

//
// Passes
//

using MemKindCreateFunc = std::function<std::optional<VPU::MemoryKind>(StringRef)>;
using ConditionFunc = std::function<bool(mlir::Operation*)>;

template <typename T>
bool isOp(mlir::Operation* op) {
    return mlir::isa<T>(op);
}

static ConditionFunc makeStubCondition() {
    return &vpux::VPUIP::isOp<IE::MemPermuteOp>;
}

std::unique_ptr<mlir::Pass> createConvertWeightsTableOp2ConstPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUpdateSwKernelParamsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDumpStatisticsOfTaskOpsPass(Logger log = Logger::global(), bool forceLogging = true);
std::unique_ptr<mlir::Pass> createUnrollClusterTilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnwrapClusterTilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollSwKernelPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDMABarrierOptimizationPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createFuseConstantsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createResolveDMAWithSwizzlingPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createMovePureViewOpBeforeCopyPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeCopiesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeConcatViewCopiesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeSubviewCopiesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeParallelCopiesPass(bool enableOptimizeConstCopy = true,
                                                             Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseLastCopyPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMoveSubViewBeforeSparseBufferPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCopyOpTilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSetMemorySpacePass(MemKindCreateFunc memKindCb, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertEltwiseToInPlacePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertSprLUTToConstPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createInsertCopyForEltwiseInPlaceInputPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLinearizationPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createBreakDataFlowPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPatchWeightsTablePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPatchPopulateWeightTableWithShavePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDMATaskProfilingReserveMemPass(
        DMAProfilingMode dmaProfilingMode = DMAProfilingMode::DISABLED, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDMATaskProfilingAfterBarrierSchedPass(
        DMAProfilingMode dmaProfilingMode = DMAProfilingMode::DISABLED, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCaptureWorkpointPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createDPUProfilingPass(MemKindCreateFunc memKindCb, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createGroupProfilingBuffersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createActShaveProfilingPass(MemKindCreateFunc memKindCb, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWrapWithPermuteAsNNDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeTileOpAsNNDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertExpandPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSwizzlingPass(const bool enableWeightSwizzling = true,
                                                const bool enableActivationSwizzling = true,
                                                Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOperationStubbingPass(ConditionFunc condition = makeStubCondition(),
                                                        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvWeightsCompressionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUngroupSparseBuffersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateSparsityCompressionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFlattenSparseWeightsTypesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createComputeSESizesPass(std::optional<bool> onlyInputsConcatOverC = std::nullopt,
                                                     Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createComputeSEBasePtrsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertSETablesToConstantsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustInputDataForExplicitSETablePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createTileActShaveKernelTaskPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSetZeroOffsetWeightsTablePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSegmentHalosPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAdjustSpillSizePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCompressDmaReserveMemPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSWKernelPrefetchingReserveMemPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createFuseDDRCopiesIntoConcats(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createLegalizeRepeatingFuncCallsPass(Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createAddCopyBetweenSWKernelsAndNetworkIOPass(Logger log = Logger::global());

//
// Asynchronous Scheduling pipeline
//

void buildAsyncSchedulingPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createWrapIntoAsyncRegionsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMoveWaitResultToAsyncBlockArgsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCalculateAsyncRegionCycleCostPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createGroupAsyncExecuteOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMoveViewOpsIntoAsyncRegionsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeAsyncDepsPass(Logger log = Logger::global());

//
// Hardware Adaptation pipeline
//

void buildHardwareAdaptationPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

std::unique_ptr<mlir::Pass> createConvertTransferOpsToDMAsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertAllocationsToDeclarationsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertFuncArgsToDeclarationsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertViewOpsToDeclarationsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertAsyncOpsToTasksPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createCompressWeightsBTCPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createNNDMATilingPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUngroupBoundedBuffersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUngroupBoundedBuffersAsFuncArgsPass(Logger log = Logger::global());

//
// Memory allocation pipeline
//

std::unique_ptr<mlir::Pass> createFeasibleAllocationPass(
        MemKindCreateFunc memKindCb, MemKindCreateFunc secondLvlMemKindCb = nullptr,
        const bool linearizeSchedule = false, const bool enablePipelining = true, const bool enablePrefetching = true,
        const bool optimizeFragmentation = true, const bool optimizeDynamicSpilling = true,
        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createQueryArgsAllocationAnalysisPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createStaticAllocationPass(MemKindCreateFunc memKindCb, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createBatchMatMulToMatMulPass(Logger log = Logger::global());

//
// DMA Unrolling Pipeline
//

void buildDMAUnrollingPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollDMAAnalysisPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollDepthToSpaceDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollSpaceToDepthDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollUpsamplingDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollPermuteToNNDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollExpandDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollPerAxisTileDMAPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createInvalidateUnrollDMAAnalysisPass(Logger log = Logger::global());

//
// DefaultHWOptions(for all devices)
//

struct DefaultHWOptionsDialectBase : public virtual vpux::DefaultHWOptionsBase {
    BoolOption enableDPUProfiling{*this, "dpu-profiling", llvm::cl::desc("Enable DPU task profiling"),
                                  llvm::cl::init(true)};

    BoolOption enableSWProfiling{*this, "sw-profiling", llvm::cl::desc("Enable SW task profiling"),
                                 llvm::cl::init(true)};

    BoolOption enableM2IProfiling{*this, "m2i-profiling", llvm::cl::desc("Enable M2I task profiling"),
                                  llvm::cl::init(true)};

    BoolOption enableOptimizeCopies{*this, "optimize-copies", llvm::cl::desc("Enable optimize-copies pass"),
                                    llvm::cl::init(true)};

    BoolOption enableOptimizeConstCopies{*this, "optimize-const-copies", llvm::cl::desc("Enable optimize-const-copies"),
                                         llvm::cl::init(true)};

    BoolOption enableGroupAsyncExecuteOps{*this, "group-async-execute-ops",
                                          llvm::cl::desc("Enable group-async-execute-ops pass"), llvm::cl::init(false)};

    BoolOption enableConstantFusion{*this, "constant-fusion", llvm::cl::desc("Enable constant fusion"),
                                    llvm::cl::init(true)};

    BoolOption enableOpsAsDMA{*this, "enable-ops-as-dma",
                              llvm::cl::desc("Force using DMA transformations instead of SW ops"),
                              llvm::cl::init(true)};

    BoolOption optimizeFragmentation{*this, "optimize-fragmentation",
                                     ::llvm::cl::desc("Enables compiler to optimize CMX fragmentation"),
                                     ::llvm::cl::init(true)};

    BoolOption optimizeDynamicSpilling{*this, "optimize-dynamic-spilling",
                                       ::llvm::cl::desc("Enables compiler to optimize dynamic spilling DMAs"),
                                       ::llvm::cl::init(true)};

    BoolOption linearizeSchedule{*this, "linearize-schedule", llvm::cl::desc("Linearize tasks on all engines"),
                                 llvm::cl::init(false)};

    BoolOption enableDumpTaskStats{*this, "dump-task-stats",
                                   ::llvm::cl::desc("Enable dumping statistics of Task operations"),
                                   ::llvm::cl::init(vpux::isDeveloperBuild())};

    BoolOption enableShaveKernelTiling{*this, "enable-shave-kernel-tiling",
                                       ::llvm::cl::desc("Enable shave kernel tiling"), ::llvm::cl::init(true)};
};

//
// Registration
//

void registerVPUIPPipelines();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/dialect/VPUIP/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/dialect/VPUIP/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace VPUIP
}  // namespace vpux
