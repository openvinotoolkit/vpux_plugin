//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::IRDFTOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                         mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                         mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                         mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));
    VPU::IRDFTOpAdaptor op(operands, attrs, prop);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }
    auto axes = parseIntArrayAttr<int64_t>(op.getAxesAttr());
    auto signalSize = parseIntArrayAttr<int64_t>(op.getSignalSizeAttr());
    const auto inType = op.getInput().getType().cast<vpux::NDTypeInterface>();
    auto outShape = to_small_vector(inType.getShape());
    // delete last size that contain complex number representation
    outShape.pop_back();
    const auto last_axis = axes.back();
    outShape[last_axis] = (outShape[last_axis] - 1) * 2;

    for (size_t i = 0; i < axes.size(); ++i) {
        if (signalSize[i] != -1) {
            outShape[axes[i]] = signalSize[i];
        }
    }

    auto outType = inType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);
    return mlir::success();
}
