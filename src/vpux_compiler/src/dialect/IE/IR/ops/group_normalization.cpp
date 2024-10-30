//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

using namespace vpux;

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::GroupNormalizationOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::GroupNormalizationOpAdaptor groupNorm(operands, attrs, prop);

    if (mlir::failed(groupNorm.verify(loc))) {
        return mlir::failure();
    }

    const auto dataType = groupNorm.getInput().getType().cast<mlir::ShapedType>();
    inferredReturnShapes.emplace_back(dataType.getShape(), dataType.getElementType());

    return mlir::success();
}
