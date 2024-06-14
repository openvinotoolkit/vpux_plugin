//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/Support/LogicalResult.h>
#include <cstdint>
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::PermuteCastOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                               std::optional<mlir::Location> optLoc,
                                                               mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                               mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::PermuteCastOpAdaptor permuteCast(operands, attrs);
    if (mlir::failed(permuteCast.verify(loc))) {
        return mlir::failure();
    }

    const auto inOrder = DimsOrder::fromValue(permuteCast.getInput());
    const auto inShape = getShape(permuteCast.getInput());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);
    if (!isTrivialPermute(inMemShape, permuteCast.getMemPerm())) {
        return errorAt(loc, "Operation represents non trivial permutation");
    }

    VPU::inferPermuteReturnTypes(permuteCast.getInput(), permuteCast.getMemPerm(), permuteCast.getDstOrder(),
                                 inferredReturnTypes);

    return mlir::success();
}

//
// DistributedCastOpInterface
//

mlir::FailureOr<VPU::DistributedTypeInterface> vpux::VPU::PermuteCastOp::inferCastedDistOutput(
        VPU::DistributedTensorType inDistributedType) {
    if (inDistributedType == nullptr || inDistributedType.getDistribution() == nullptr) {
        return mlir::failure();
    }
    const auto ctx = getContext();
    const auto origDistribution = inDistributedType.getDistribution();
    const auto srcType = getInput().getType().cast<NDTypeInterface>();
    const auto dstType = getOutput().getType().cast<NDTypeInterface>();

    auto castedOutputDistribution =
            applyPermutationOnDistributedTensorAttr(origDistribution, getMemPerm(), srcType.getDimsOrder(),
                                                    dstType.getDimsOrder(), srcType.getShape(), dstType.getShape());
    if (isSegmentedLikeMode(inDistributedType)) {
        auto legalizeDistribution = legalizeCastedDistribution(castedOutputDistribution, ctx);
        if (mlir::failed(legalizeDistribution)) {
            return mlir::failure();
        }

        castedOutputDistribution = legalizeDistribution.value();
    }

    const auto dstDimsOrderAttr = mlir::AffineMapAttr::get(dstType.getDimsOrder().toAffineMap(ctx));
    const auto newDistributionType =
            DistributedTensorType::get(ctx, dstType.getShape().raw(), dstType.getElementType(), dstDimsOrderAttr,
                                       inDistributedType.getMemSpace(), castedOutputDistribution);
    return newDistributionType.cast<VPU::DistributedTypeInterface>();
}
