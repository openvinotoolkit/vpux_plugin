//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPURegMapped/passes.hpp"

#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPURT/IR/dialect.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace vpux {
namespace VPUMI40XX {

//
// Passes
//

std::unique_ptr<mlir::Pass> createSetupProfilingVPUMI40XXPass(
        DMAProfilingMode dmaProfilingMode = DMAProfilingMode::DISABLED, Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createBarrierComputationPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> reorderMappedInferenceOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createResolveTaskLocationPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createBarrierTopologicalMappingPass(const int barrierThreshold = 3000,
                                                                Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createGroupExecutionOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnGroupExecutionOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createWorkloadManagementPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createResolveWLMTaskLocationPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPropagateFinalBarrierPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddEnqueueOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollFetchTaskOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLinkEnqueueTargetsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createLinkAllOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollEnqueueOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSplitEnqueueOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddBootstrapOpsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createNextSameIdAssignmentPass(Logger log = Logger::global());

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/dialect/VPUMI40XX/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace VPUMI40XX
}  // namespace vpux
