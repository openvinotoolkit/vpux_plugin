//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

namespace vpux {
namespace IE {
bool beneficialToKeepExpand(ShapeRef unExpandedShape, ShapeRef expandedShape, mlir::Operation* op);

// convert expand op to convolution utils
int64_t calculateAlignmentRequirementForExpandOpConversion(const vpux::NDTypeInterface expandInType);
bool isEligibleConvertToConv(IE::ExpandOp expandOp, Logger log, StringRef debugName);
std::optional<vpux::Dim> getExpandAxis(IE::ExpandOp expandOp);
}  // namespace IE
}  // namespace vpux
