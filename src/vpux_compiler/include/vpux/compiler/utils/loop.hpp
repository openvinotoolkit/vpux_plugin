//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/MLIRContext.h>

#include "vpux/utils/core/func_ref.hpp"

namespace vpux {
enum class LoopExecPolicy {
    Sequential,
    Parallel,
};

void loop_1d(LoopExecPolicy policy, mlir::MLIRContext* ctx, int64_t dim0, vpux::FuncRef<void(int64_t)> func);

void loop_2d(LoopExecPolicy policy, mlir::MLIRContext* ctx, int64_t dim0, int64_t dim1,
             FuncRef<void(int64_t, int64_t)> func);

void loop_3d(LoopExecPolicy policy, mlir::MLIRContext* ctx, int64_t dim0, int64_t dim1, int64_t dim2,
             FuncRef<void(int64_t, int64_t, int64_t)> func);

void loop_4d(LoopExecPolicy policy, mlir::MLIRContext* ctx, int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
             FuncRef<void(int64_t, int64_t, int64_t, int64_t)> func);
}  // namespace vpux
