//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

namespace vpux {
namespace IE {

// This function will/must be removed when tiling support for NCE.Matmul is implemented (E125519)
bool doesIEMatMulFitIntoCMX(IE::MatMulOp matmulOp, ShapeRef input1Shape, ShapeRef input2Shape);

bool isMatmulWithRHSTransposition(IE::MatMulOp matmulOp);

}  // namespace IE
}  // namespace vpux
