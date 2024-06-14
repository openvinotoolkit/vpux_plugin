//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux {
namespace IE {

mlir::FailureOr<SmallVector<int64_t>> extractPads(mlir::ArrayAttr padValue, Logger log);
mlir::FailureOr<SmallVector<int64_t>> extractPads(mlir::Location loc, const mlir::Value& padValue,
                                                  const std::optional<mlir::ArrayAttr>& padAttr,
                                                  vpux::ShapeRef inputShape);

}  // namespace IE
}  // namespace vpux
