//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <numeric>
#include "vpux/compiler/dialect/IE/IR/ops.hpp"

namespace vpux {
namespace IE {

struct RollShiftAndAxes final {
    SmallVector<int64_t> shift;
    SmallVector<int64_t> axes;

    explicit RollShiftAndAxes(ArrayRef<int64_t> shiftRef, ArrayRef<int64_t> axesRef): shift(shiftRef), axes(axesRef) {
    }
};

mlir::FailureOr<RollShiftAndAxes> getShiftAndAxesForRollOp(mlir::Location loc, mlir::Value shiftValue,
                                                           mlir::Value axesValue, ShapeRef inputShape);

}  // namespace IE
}  // namespace vpux
