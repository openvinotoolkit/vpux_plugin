//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

using namespace vpux;

mlir::Value VPUIP::StubOp::getViewSource() {
    // Limitation:
    // StubOp is a generic placeholder for operations, with a variadic number of input and output buffers.
    // In the case of ViewLikeOpInterface, getViewSource returns the associated input buffer for the single output
    // buffer, but this doesn't fully cover our Stub Operation, which supports any combination of inputs and outputs.
    // For now, we use this approach.
    return getInputs()[0];
}
