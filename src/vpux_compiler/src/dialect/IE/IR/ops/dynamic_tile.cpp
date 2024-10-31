//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::DynamicTileOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::DynamicTileOpAdaptor tile(operands, attrs, prop);
    if (mlir::failed(tile.verify(loc))) {
        return mlir::failure();
    }

    if (tile.getRepeats() != nullptr && tile.getRepeatsValues().has_value()) {
        return errorAt(loc, "Ambiguous repeats representation");
    }

    const auto outShape = parseIntArrayAttr<int64_t>(tile.getOutputShape());
    const auto outBounds = tile.getOutputBoundsAttr();

    const auto inType = tile.getInput().getType().cast<mlir::RankedTensorType>();

    const auto outDesc =
            vpux::getTensorAttr(ctx, DimsOrder::fromNumDims(outShape.size()), vpux::getMemorySpace(inType), outBounds);
    inferredReturnShapes.emplace_back(outShape, inType.getElementType(), outDesc);

    return mlir::success();
}
