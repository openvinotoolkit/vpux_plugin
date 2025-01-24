//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

namespace vpux::vpuip2vpumi40xx {

mlir::SmallVector<mlir::Value> convertOrUnrollBuffer(mlir::OpBuilder builder, mlir::Value output);
mlir::Value convertOrExtractBuffer(mlir::OpBuilder builder, mlir::Value output, uint32_t tileIndex);

}  // namespace vpux::vpuip2vpumi40xx
