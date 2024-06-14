//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

namespace vpux {
namespace IE {

mlir::LogicalResult isBeneficialConvertScaleShiftToDW(IE::ScaleShiftOp scaleShiftOp, Logger log);

}  // namespace IE
}  // namespace vpux
