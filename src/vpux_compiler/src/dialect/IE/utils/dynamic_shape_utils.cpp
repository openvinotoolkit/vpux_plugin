//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"

namespace vpux::IE {

bool hasDynamicShape(const mlir::Value value) {
    return getShape(value).isDynamic();
}

bool hasDynamicTensors(mlir::Operation* op) {
    const auto hasDynamicInputs = llvm::any_of(op->getOperands(), hasDynamicShape);
    const auto hasDynamicOutputs = llvm::any_of(op->getResults(), hasDynamicShape);

    return hasDynamicInputs || hasDynamicOutputs;
}

// The list of operations may seem completely arbitrary, but they share one common trait.
// Convolution, MaxPool, Add and ReLU all run on static shapes only.
// In most cases these operations are mapped to DPU.
// DPU cannot adjust the workload size once it is set during the parsing stage.
// Right now these operations use upper bounds to set the size of workloads.
bool needsStaticShape(mlir::Operation* op) {
    return mlir::isa_and_nonnull<ShapeBoundOp>(op);
}

}  // namespace vpux::IE
