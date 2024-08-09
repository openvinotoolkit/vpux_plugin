//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"
#include "vpux/compiler/core/type_interfaces.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux::VPU {

/*
   Find right class to get strategies for particular platform
*/
std::unique_ptr<IGreedilyPassStrategy> createMakeOpsWithDistributedTensorStrategyGetter(
        mlir::func::FuncOp funcOp, llvm::DenseMap<mlir::OpResult, OverlapDistributionParams>& overlapParamsLookup,
        bool enableExplicitDistributedTensorAttr);

}  // namespace vpux::VPU
