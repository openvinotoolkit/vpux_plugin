//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/VPU30XX/core/pipelines_options.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

namespace vpux {
namespace IE {
namespace arch30xx {

//
// Passes
//

std::unique_ptr<mlir::Pass> createConvertTile2PerAxisTilePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createInsertIdentityPoolBeforeOpPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createMapBilinearInterpolateOnDPUPass(const bool interpolateAsSEOp = false,
                                                                  Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createOptimizeSliceExpandPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createExpandActivationChannelsPass(const bool seOpsEnabled = false,
                                                               const bool seExperimentalOpsEnabled = false,
                                                               Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnrollBatchPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createConvertToMixedPrecision(Logger log = Logger::global());

//
// Pipelines
//

void buildInitialTransformationsPipeline(mlir::OpPassManager& pm, const TransformOptions& options,
                                         Logger log = Logger::global());
void buildOptimizeActivationsPipeline(mlir::OpPassManager& pm, const OptimizeActivationsOptions& options,
                                      Logger log = Logger::global());
void buildMemPermuteProcessingPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());
void buildLowPrecisionPipeline(mlir::OpPassManager& pm, const LowPrecisionOptions& options,
                               Logger log = Logger::global());

//
// AdjustLayout
//

void buildAdjustLayoutPipeline(mlir::OpPassManager& pm, const AdjustLayoutOptions& options,
                               Logger log = Logger::global());

//
// DefaultHWOptions
//
struct DefaultHWOptions : public IE::DefaultHWOptionsDialectBase, virtual vpux::arch30xx::DefaultHWOptionsDeviceBase {};

void buildDefaultHWPipeline(mlir::OpPassManager& pm, const DefaultHWOptions& options, Logger log = Logger::global());

//
// registerIEPipelines
//

void registerIEPipelines();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/VPU30XX/dialect/IE/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/VPU30XX/dialect/IE/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace arch30xx
}  // namespace IE
}  // namespace vpux
