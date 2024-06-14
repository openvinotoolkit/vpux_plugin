//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

namespace vpux {
namespace IE {

DimArr getDiffInOutSizeDims(ShapeRef inShape, ShapeRef outShape);
std::optional<vpux::Dim> getSingleDiffAxis(ShapeRef inShape, ShapeRef outShape);

}  // namespace IE
}  // namespace vpux
