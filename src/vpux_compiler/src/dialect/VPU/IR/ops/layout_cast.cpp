//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::LayoutCastOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              std::optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::OpaqueProperties prop,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::LayoutCastOpAdaptor overrideLayout(operands, attrs, prop);
    if (mlir::failed(overrideLayout.verify(loc))) {
        return mlir::failure();
    }

    const auto outAffineMap = overrideLayout.getDstOrder();
    const auto inType = overrideLayout.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto outType = inType.changeDimsOrder(DimsOrder::fromAffineMap(outAffineMap));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// verify
//

mlir::LogicalResult vpux::VPU::LayoutCastOp::verify() {
    const auto outAffineMap = getDstOrder();
    const auto inType = getInput().getType().cast<vpux::NDTypeInterface>();
    if (inType.getRank() != outAffineMap.getNumDims()) {
        return errorAt(*this, "Cannot apply {0} map to {1}.", outAffineMap, inType.getShape());
    }

    return mlir::success();
}

//
// DistributedCastOpInterface
//

mlir::FailureOr<VPU::DistributedTypeInterface> vpux::VPU::LayoutCastOp::inferCastedDistOutput(
        VPU::DistributedTensorType inDistributedType) {
    if (inDistributedType == nullptr || inDistributedType.getDistribution() == nullptr) {
        return mlir::failure();
    }
    const auto ctx = getContext();
    const auto srcType = getInput().getType().cast<NDTypeInterface>();
    const auto dstType = getOutput().getType().cast<NDTypeInterface>();
    const auto srcOrder = srcType.getDimsOrder();
    const auto dstOrder = dstType.getDimsOrder();
    const auto memPerm = getPermutationFromOrders(srcOrder, dstOrder, ctx);

    auto castedOutputDistribution =
            applyPermutationOnDistributedTensorAttr(inDistributedType, memPerm, srcType.getDimsOrder(),
                                                    dstType.getDimsOrder(), srcType.getShape(), dstType.getShape());
    if (mlir::failed(castedOutputDistribution)) {
        return mlir::failure();
    }

    const auto dstDimsOrderAttr = mlir::AffineMapAttr::get(dstType.getDimsOrder().toAffineMap(ctx));
    const auto newDistributionType =
            DistributedTensorType::get(ctx, dstType.getShape().raw(), dstType.getElementType(), dstDimsOrderAttr,
                                       inDistributedType.getMemSpace(), castedOutputDistribution.value());
    return newDistributionType.cast<VPU::DistributedTypeInterface>();
}
