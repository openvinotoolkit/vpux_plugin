//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"

#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/core/logger.hpp"

namespace vpux {
namespace arch30xx {

//
// LowerIE2VPU
//

std::unique_ptr<mlir::Pass> createConvertIEToVPUNCEPass(Logger log = Logger::global());

//
// pipelines
//

void buildLowerIE2VPUPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());
void buildLowerVPU2VPUIPPipeline(mlir::OpPassManager& pm, Logger log = Logger::global());

//
// registerConversionPipeline
//

void registerConversionPipeline();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/VPU30XX/conversion/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/VPU30XX/conversion/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace arch30xx
}  // namespace vpux
