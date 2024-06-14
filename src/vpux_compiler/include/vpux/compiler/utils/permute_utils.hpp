//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"

namespace vpux {

MemShape applyPerm(MemShapeRef memShape, mlir::AffineMap memPerm);

SmallVector<int64_t> getPermutateDims(MemShapeRef inShape, mlir::AffineMap memPerm);
bool isTrivialPermute(MemShapeRef inShape, mlir::AffineMap memPerm);
bool isTrivialReorder(DimsOrder inOrder, DimsOrder outOrder, ShapeRef shape);
bool isTrivialReorder(IE::ReorderOp origOp);

mlir::AffineMap getPermutationFromOrders(DimsOrder inOrder, DimsOrder outOrder, mlir::MLIRContext* ctx);
DimsOrder applyPermutation(const DimsOrder lhs, const DimsOrder rhs);

VPU::DistributedTensorAttr applyPermutationOnDistributedTensorAttr(VPU::DistributedTensorAttr inDistribution,
                                                                   mlir::AffineMap memPerm, DimsOrder srcOrder,
                                                                   DimsOrder dstOrder, ShapeRef srcShape,
                                                                   ShapeRef dstShape);
DimsOrder moveD0ToTheFront(DimsOrder inOrder);

std::pair<SmallVector<uint32_t>, SmallVector<int64_t>> getMergedPermutationAndShape(NDTypeInterface input,
                                                                                    mlir::AffineMap permutation,
                                                                                    int64_t rank = 4);
void extendPermutationAndShape(SmallVector<uint32_t>& permutation, SmallVector<int64_t>& shape, int64_t rank);

IE::LayerWithPermuteInterface getFusableLayerWithPermuteInterface(mlir::Operation* op);

}  // namespace vpux
