//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"

namespace vpux {
namespace VPURT {
namespace arch37xx {

//
// Passes
//

std::unique_ptr<mlir::Pass> createAddUpdateBarrierForSwKernelsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddFinalBarrierPass(Logger log = Logger::global());

//
// Generated
//

#define GEN_PASS_CLASSES
#include <vpux/compiler/NPU37XX/dialect/VPURT/passes.hpp.inc>
#undef GEN_PASS_CLASSES

#define GEN_PASS_REGISTRATION
#include <vpux/compiler/NPU37XX/dialect/VPURT/passes.hpp.inc>
#undef GEN_PASS_REGISTRATION

}  // namespace arch37xx
}  // namespace VPURT
}  // namespace vpux
