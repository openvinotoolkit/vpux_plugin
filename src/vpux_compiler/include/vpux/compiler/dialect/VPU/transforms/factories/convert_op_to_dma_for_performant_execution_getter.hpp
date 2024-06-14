//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux::VPU {

/*
   Find right class to get strategies for particular platform
*/
std::unique_ptr<IConversionPassStrategy> CreateConvertOpToDMAForPerformantExecutionStrategy(ArchKind arch);

}  // namespace vpux::VPU
