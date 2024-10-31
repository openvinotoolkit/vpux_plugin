//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"

namespace vpux::IE {

bool hasDynamicTensors(mlir::Operation* op) {
    const auto isDynamic = [](mlir::Value value) {
        return getShape(value).isDynamic();
    };

    const auto hasDynamicInputs = llvm::any_of(op->getOperands(), isDynamic);
    const auto hasDynamicOutputs = llvm::any_of(op->getResults(), isDynamic);

    return hasDynamicInputs || hasDynamicOutputs;
}

}  // namespace vpux::IE
