//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::GatherElementsOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GatherElementsOpAdaptor gatherElements(operands, attrs, prop);
    if (mlir::failed(gatherElements.verify(loc))) {
        return mlir::failure();
    }

    const auto inIndicesType = gatherElements.getIndices().getType().cast<vpux::NDTypeInterface>();
    const auto inInputType = gatherElements.getInput().getType().cast<vpux::NDTypeInterface>();

    const auto outType = inInputType.changeShape(inIndicesType.getShape());
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
