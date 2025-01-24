//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUASM/dialect.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/types.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/optional.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace vpux {
namespace VPUASM {

//
// Passes
//

std::unique_ptr<mlir::Pass> createHoistInputOutputsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddProfilingSectionPass(Logger log = Logger::global());
//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/dialect/VPUASM/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/dialect/VPUASM/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace VPUASM
}  // namespace vpux
