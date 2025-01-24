//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/Pass/Pass.h>

namespace vpux {
namespace Const {

//
// Passes
//

std::unique_ptr<mlir::Pass> createConstantFoldingPass(Logger log = Logger::global(),
                                                      const int64_t threshold = 300 * 1024 * 1024);  // 300MB
std::unique_ptr<mlir::Pass> createApplySwizzlingPass();

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/dialect/const/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/dialect/const/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace Const
}  // namespace vpux
