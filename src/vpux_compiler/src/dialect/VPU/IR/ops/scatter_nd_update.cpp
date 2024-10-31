//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ScatterNDUpdateOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ScatterNDUpdateOpAdaptor scatter(operands, attrs, prop);
    if (mlir::failed(scatter.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = scatter.getInput().getType();
    const auto outShape = inType.cast<vpux::NDTypeInterface>().getShape().toValues();

    auto outType = inType;
    if (outShape.isDynamic()) {
        const auto inputBoundsAttr = vpux::getBounds(outType.cast<mlir::RankedTensorType>());
        const auto bounds = parseIntArrayAttr<int64_t>(inputBoundsAttr);
        if (bounds.empty()) {
            return errorAt(loc, "VPU::ScatterNDUpdateOp::inferReturnTypes. Got empty bounds array");
        }

        outType = outType.cast<vpux::BoundedTypeInterface>().changeBounds(inputBoundsAttr);
    }
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
