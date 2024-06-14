//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

namespace vpux {
namespace IE {

void handleConsecutiveOnes(ArrayRef<int64_t> inShape, ArrayRef<int64_t> outShape, std::size_t& startIn,
                           std::size_t& startOut, SmallVector<SmallVector<int64_t>>& reassociationVec);
// Note: When having dims equal to 1 in one of the shapes that do not have a corresponding 1 in the other shape, there
// might be multiple dim associations possible.
// E.g.: 1 x 2 x 2 x 1 x 2 x 3 -> 1 x 4 x 6 has 2 possible mappings:
//      {0} -> {0}, {1, 2, 3} -> {1}, {4, 5} -> {2} (this one is computed by the fcn below)
//      {0} -> {0}, {1, 2} -> {1}, {3, 4, 5} -> {2} (this one is computed by the extended fcn below)
mlir::FailureOr<SmallVector<SmallVector<int64_t>>> getReassociationMap(ArrayRef<int64_t> inShape,
                                                                       ArrayRef<int64_t> outShape);
mlir::FailureOr<SmallVector<SmallVector<int64_t>>> getReassociationMapExtension(ArrayRef<int64_t> inShape,
                                                                                ArrayRef<int64_t> outShape);

bool isNotDimExpansionReshape(ShapeRef origShape, ShapeRef reshapeShape);
bool isNotDimShrinkReshape(ShapeRef origShape, ShapeRef reshapeShape);

IE::ShapeCastOp buildShapeCast(mlir::Location loc, mlir::Value input, ArrayRef<int64_t> targetShape,
                               mlir::PatternRewriter& rewriter);
}  // namespace IE
}  // namespace vpux
