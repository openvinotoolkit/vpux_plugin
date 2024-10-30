//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace vpux {
namespace VPURT {

using MemoryEffect = mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>;
void getTaskEffects(mlir::Operation* op, SmallVectorImpl<MemoryEffect>& effects);

}  // namespace VPURT
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/VPURT/ops_interfaces.hpp.inc>
