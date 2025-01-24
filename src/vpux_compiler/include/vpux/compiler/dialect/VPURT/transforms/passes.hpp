//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace vpux {
namespace VPURT {

//
// Barrier Legalization Pipeline
//

void buildBarrierLegalizationPipeline(mlir::OpPassManager& pm,
                                      std::optional<int> virtualBarrierThresholdforWlm = std::nullopt,
                                      const bool unevenVariantSplitFlag = false, Logger log = Logger::global());

//
// Passes
//

std::unique_ptr<mlir::Pass> createSplitControlGraphPass(
        const int controlGraphSplitBlockSize = CONTROL_GRAPH_SPLIT_BLOCK_SIZE, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSimplifySchedulePass(const bool shareWaitAndUpdateBarriersFlag = true,
                                                       const bool reduceParallelControlFlowsFlag = true,
                                                       Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitExceedingVariantCountBarriersPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSatisfyOneWaitBarrierPerTaskPass(
        std::optional<int> virtualBarrierThresholdforWlm = std::nullopt, const bool unevenVariantSplitFlag = false,
        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createReduceExceedingActiveCountBarriersPass(
        std::optional<int> virtualBarrierThresholdforWlm = std::nullopt, const bool UnevenVariantSplitFlag = false,
        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAssignPhysicalBarriersPass(
        const bool barrierColorBinFlag = false, std::optional<int> virtualBarrierThresholdforWlm = std::nullopt,
        Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createBarrierSimulationPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createIntermediateBufferOutputPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createInferenceExecutionAnalysisPass(
        std::string compileSchedTraceFileName = "compileTimeScheduleTrace.json", bool dumpToJson = false,
        bool enableActivityFactor = true, Logger log = Logger::global());

//
// Registration
//

void registerVPURTPipelines();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/dialect/VPURT/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/dialect/VPURT/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace VPURT
}  // namespace vpux
