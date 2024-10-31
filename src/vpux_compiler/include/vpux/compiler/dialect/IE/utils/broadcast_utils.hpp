//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <numeric>
#include "vpux/compiler/dialect/IE/IR/ops.hpp"

namespace vpux {
namespace IE {

SmallVector<int64_t> getBroadcastAxesNumpyBidirectional(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> outputShape);
SmallVector<int64_t> getBroadcastAxesExplicit(ArrayRef<int64_t> axesMapping, ArrayRef<int64_t> outputShape);
mlir::Value createShapeConstForBroadCast(mlir::PatternRewriter& rewriter, mlir::MLIRContext* ctx, mlir::Location loc,
                                         ShapeRef shape);

/// Aligns shapes of two buffers by broadcasting across misaligned axes in order
/// to allow 'X op Y' operation to behave correctly. This procedure could fail
/// when x's and y's shapes differ and both are non-1 at any given axis (this
/// situation is ambiguous).
///
/// Note: this procedure does not guarantee any state consistency if it fails
/// "mid-way", that is, any operands (x or y) could still be modified (e.g.
/// broadcasted across some axes).
mlir::LogicalResult broadcastAlignShapes(mlir::MLIRContext* ctx, Const::Content& x, Const::Content& y,
                                         const Logger& log);
}  // namespace IE
}  // namespace vpux
