//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"

namespace vpux {

constexpr int64_t PERMUTE_TO_POOLING_THRESHOLD = 32 * 16 * 224;

MemShape applyPerm(MemShapeRef memShape, mlir::AffineMap memPerm);

SmallVector<int64_t> getPermutateDims(MemShapeRef inShape, mlir::AffineMap memPerm);
bool isTrivialPermute(MemShapeRef inShape, mlir::AffineMap memPerm);
bool isTrivialReorder(DimsOrder inOrder, DimsOrder outOrder, ShapeRef shape);
bool isTrivialReorder(IE::ReorderOp origOp);

mlir::AffineMap getPermutationFromOrders(DimsOrder inOrder, DimsOrder outOrder, mlir::MLIRContext* ctx);
DimsOrder applyPermutation(const DimsOrder lhs, const DimsOrder rhs);

template <typename T, std::enable_if_t<or_<std::is_same<VPU::DistributedTensorType, T>,
                                           std::is_same<VPUIP::DistributedBufferType, T>>::value,
                                       bool> = true>
mlir::FailureOr<VPU::DistributionInfoAttr> applyPermutationOnDistributionInfoAttr(T inDistributedType,
                                                                                  mlir::AffineMap memPerm,
                                                                                  DimsOrder srcOrder,
                                                                                  DimsOrder dstOrder, ShapeRef srcShape,
                                                                                  ShapeRef dstShape) {
    const auto inDistribution = VPU::DistributionInfo::getClassFromAttr(inDistributedType.getDistribution());

    auto distributionInfoOrFailure = applyPermutationOnDistributionInfo(inDistributedType, inDistribution, memPerm,
                                                                        srcOrder, dstOrder, srcShape, dstShape);
    if (mlir::failed(distributionInfoOrFailure)) {
        return mlir::failure();
    }

    return VPU::DistributionInfo::getAttrFromClass(inDistributedType.getContext(), distributionInfoOrFailure.value());
}

mlir::FailureOr<VPU::DistributionInfo> applyPermutationOnDistributionInfo(vpux::NDTypeInterface inType,
                                                                          const VPU::DistributionInfo& inDistribution,
                                                                          mlir::AffineMap memPerm, DimsOrder srcOrder,
                                                                          DimsOrder dstOrder, ShapeRef srcShape,
                                                                          ShapeRef dstShape);

DimsOrder moveD0ToTheFront(DimsOrder inOrder);

std::pair<SmallVector<uint32_t>, SmallVector<int64_t>> getMergedPermutationAndShape(NDTypeInterface input,
                                                                                    mlir::AffineMap permutation,
                                                                                    int64_t rank = 4);
void extendPermutationAndShape(SmallVector<uint32_t>& permutation, SmallVector<int64_t>& shape, int64_t targetRank);

IE::LayerWithPermuteInterface getFusableLayerWithPermuteInterface(mlir::Operation* op);

NDTypeInterface inferNewTypeWithMemPerm(NDTypeInterface oldType, mlir::AffineMap memPerm, const DimsOrder& dstOrder);

std::optional<IE::PermuteCastOp> tryToFindPermuteCastOp(mlir::Location loc, mlir::Value input, DimsOrder outOrder,
                                                        ShapeRef outShape, mlir::PatternRewriter& rewriter);

Dim inferDimAfterPermutation(Dim dim, DimsOrder srcOrder, DimsOrder dstOrder, mlir::AffineMap perm);
}  // namespace vpux
