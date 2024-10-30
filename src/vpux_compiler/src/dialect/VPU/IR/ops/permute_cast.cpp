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
                                                               mlir::OpaqueProperties prop,
                                                               mlir::RegionRange /*regions*/,
                                                               mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::PermuteCastOpAdaptor permuteCast(operands, attrs, prop);
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

mlir::FailureOr<std::pair<mlir::Type, VPU::DistributionInfo>> vpux::VPU::PermuteCastOp::inferCastedTypeAndDistribution(
        vpux::NDTypeInterface inType, VPU::DistributionInfo& distribution) {
    if (inType == nullptr || mlir::isa<VPU::DistributedTensorType>(inType) ||
        distribution.getDistributionMode() == DistributionMode::NONE) {
        return mlir::failure();
    }
    const auto srcType = getInput().getType().cast<NDTypeInterface>();
    const auto dstType = getOutput().getType().cast<NDTypeInterface>();

    auto castedOutputDistribution =
            applyPermutationOnDistributionInfo(inType, distribution, getMemPerm(), srcType.getDimsOrder(),
                                               dstType.getDimsOrder(), srcType.getShape(), dstType.getShape());
    if (mlir::failed(castedOutputDistribution)) {
        return mlir::failure();
    };
    const auto typeComponents = TypeComponents()
                                        .setShape(dstType.getShape())
                                        .setDimsOrder(dstType.getDimsOrder())
                                        .setElementType(dstType.getElementType())
                                        .setMemSpace(inType.getMemSpace());
    return std::make_pair(mlir::cast<mlir::Type>(dstType.changeTypeComponents(typeComponents)),
                          castedOutputDistribution.value());
}
