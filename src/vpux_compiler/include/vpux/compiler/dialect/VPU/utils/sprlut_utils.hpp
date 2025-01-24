//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/Operation.h>

namespace vpux {
namespace VPU {

constexpr auto SPRLUT_ALIGNMENT_REQUIREMENT = 32;

constexpr StringRef SPRLUT_ENABLED = "VPU.SprLUTEnabled";

bool isSprLUTEnabled(mlir::Operation* op);

}  // namespace VPU
}  // namespace vpux
