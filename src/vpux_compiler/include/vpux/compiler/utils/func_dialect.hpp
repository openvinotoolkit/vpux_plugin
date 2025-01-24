//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>

#include "vpux/utils/core/small_vector.hpp"

namespace vpux {

//
// getCalledFunction
//

mlir::func::FuncOp getCalledFunction(mlir::CallOpInterface callOp);

//
// getCallSites
//

SmallVector<mlir::func::CallOp> getCallSites(mlir::func::FuncOp funcOp, mlir::Operation* from);

}  // namespace vpux
