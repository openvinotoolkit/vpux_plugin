//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux {

//
// getCalledFunction
//

mlir::func::FuncOp getCalledFunction(mlir::CallOpInterface callOp);

}  // namespace vpux
