//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux {
namespace IE {

void setupConvertPrecision(mlir::TypeConverter& typeConverter, FuncRef<mlir::Type(mlir::Type)> elemTypeConversionCb);

mlir::LogicalResult runConvertPrecision(mlir::ModuleOp module, mlir::TypeConverter& typeConverter,
                                        mlir::ConversionTarget& target, Logger& log);

}  // namespace IE
}  // namespace vpux
