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
mlir::FailureOr<VPU::DistributedTensorAttr> applyPermutationOnDistributedTensorAttr(
        T inDistributedType, mlir::AffineMap memPerm, DimsOrder srcOrder, DimsOrder dstOrder, ShapeRef srcShape,
        ShapeRef dstShape) {
    auto ctx = inDistributedType.getContext();
    auto inDistribution = inDistributedType.getDistribution();

    auto permuteAxisOfArray = [&](ArrayRef<int64_t> arr) -> SmallVector<int64_t> {
        // At VPUIP level, VPU.LayoutCast gets lowered to VPUIP.PermuteCast.
        // LayoutCast will have same in/out shape but different orders, which cannot be handled
        // the same way as the VPU.PermuteCast ops which have the same memory shape between input
        // and output even if orders and logical shapes differ. In such a case, applying the
        // `toMemoryOrder -> applyPerm -> toLogicalOrder` transformations will not permute the
        // distributed attr correctly.
        if (srcShape == dstShape) {
            return SmallVector<int64_t>(arr);
        }

        const auto arrInMemOrder = srcOrder.toMemoryOrder(Shape(arr));
        const auto arrPermutedInMemOrder = applyPerm(arrInMemOrder, memPerm);
        const auto arrPermutedInLogicalOrder = dstOrder.toLogicalOrder(arrPermutedInMemOrder).raw();

        return arrPermutedInLogicalOrder;
    };

    auto numTilesAttr = inDistribution.getNumTiles();
    if (numTilesAttr != nullptr) {
        const auto numTilesVec = parseIntArrayAttr<int64_t>(numTilesAttr);
        numTilesAttr = getIntArrayAttr(ctx, permuteAxisOfArray(numTilesVec));
    }

    auto alignmentAttr = inDistribution.getAlignment();
    if (alignmentAttr != nullptr) {
        const auto alignmentVec = parseIntArrayAttr<int64_t>(alignmentAttr);
        alignmentAttr = getIntArrayAttr(ctx, permuteAxisOfArray(alignmentVec));
    }

    auto permutePerClusterShapesOffsets = [&](mlir::ArrayAttr shapesOffsetsAttr) -> mlir::ArrayAttr {
        const auto inPerClusterShapesOffsetsVec = parseIntArrayOfArrayAttr<int64_t>(shapesOffsetsAttr);
        auto outComputeShapesVec = SmallVector<SmallVector<int64_t>>();

        for (const auto& shapesOffsets : inPerClusterShapesOffsetsVec) {
            outComputeShapesVec.push_back(permuteAxisOfArray(shapesOffsets));
        }

        return getIntArrayOfArray(ctx, outComputeShapesVec);
    };

    auto computeShapesAttr = (inDistribution.getComputeShapes() != nullptr)
                                     ? permutePerClusterShapesOffsets(inDistribution.getComputeShapes())
                                     : inDistribution.getComputeShapes();

    auto computeOffsetsAttr = (inDistribution.getComputeOffsets() != nullptr)
                                      ? permutePerClusterShapesOffsets(inDistribution.getComputeOffsets())
                                      : inDistribution.getComputeOffsets();

    auto memoryShapesAttr = (inDistribution.getMemoryShapes() != nullptr)
                                    ? permutePerClusterShapesOffsets(inDistribution.getMemoryShapes())
                                    : inDistribution.getMemoryShapes();

    auto memoryOffsetsAttr = (inDistribution.getMemoryOffsets() != nullptr)
                                     ? permutePerClusterShapesOffsets(inDistribution.getMemoryOffsets())
                                     : inDistribution.getMemoryOffsets();

    auto distribution = VPU::DistributedTensorAttr::get(
            ctx, inDistribution.getMode(), numTilesAttr, inDistribution.getKernel(), inDistribution.getPads(),
            inDistribution.getStrides(), inDistribution.getNumClusters(), alignmentAttr,
            inDistribution.getUniformDistributedSegments(), computeShapesAttr, computeOffsetsAttr, memoryShapesAttr,
            memoryOffsetsAttr, inDistribution.getEqualMemoryAndComputeView());

    if (inDistribution.getMode().getValue() != VPU::DistributionMode::OVERLAPPED) {
        return distribution;
    }

    if (VPU::isOverlappedOverH(distribution) || VPU::isOverlappedOverW(distribution)) {
        return distribution;
    }

    if (VPU::isSegmentedLikeMode(inDistributedType)) {
        return VPU::legalizeCastedDistribution(distribution, ctx);
    }

    return mlir::failure();
}

DimsOrder moveD0ToTheFront(DimsOrder inOrder);

std::pair<SmallVector<uint32_t>, SmallVector<int64_t>> getMergedPermutationAndShape(NDTypeInterface input,
                                                                                    mlir::AffineMap permutation,
                                                                                    int64_t rank = 4);
void extendPermutationAndShape(SmallVector<uint32_t>& permutation, SmallVector<int64_t>& shape, int64_t targetRank);

IE::LayerWithPermuteInterface getFusableLayerWithPermuteInterface(mlir::Operation* op);

}  // namespace vpux
