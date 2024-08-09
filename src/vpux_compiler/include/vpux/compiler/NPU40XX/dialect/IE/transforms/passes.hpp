//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU40XX/core/pipelines_options.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

namespace vpux {
namespace IE {
namespace arch40xx {

//
// Passes
//

std::unique_ptr<mlir::Pass> createMapBilinearInterpolateOnDPUPass(const bool interpolateAsSEOp = false,
                                                                  Logger log = Logger::global());

//
// DefaultHWOptions
//

struct DefaultHWOptions : public IE::DefaultHWOptionsDialectBase, virtual vpux::arch40xx::DefaultHWOptionsDeviceBase {
    BoolOption enableConvertFFTToConv{*this, "convert-fft-to-conv", llvm::cl::desc("Enable convert-fft-to-conv pass"),
                                      llvm::cl::init(true)};

    BoolOption enableFusePermuteQuantize{*this, "fuse-permute-quantize",
                                         llvm::cl::desc("Enable fuse-permute-quantize pass"), llvm::cl::init(true)};

    BoolOption enableFusePermuteQuantizeExpand{*this, "fuse-permute-quantize-expand",
                                               llvm::cl::desc("Enable fuse-permute-quantize-expand pass"),
                                               llvm::cl::init(true)};
};

//
// Pipelines
//

void buildDefaultHWPipeline(mlir::OpPassManager& pm, const IE::arch40xx::DefaultHWOptions& options,
                            Logger log = Logger::global());

//
// registerIEPipelines
//

void registerIEPipelines();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/NPU40XX/dialect/IE/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/NPU40XX/dialect/IE/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace arch40xx
}  // namespace IE
}  // namespace vpux
