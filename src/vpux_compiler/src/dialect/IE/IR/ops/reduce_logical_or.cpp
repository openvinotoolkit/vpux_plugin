//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ReduceLogicalOrOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ReduceLogicalOrOpAdaptor reduceLogicalOr(operands, attrs, prop);
    if (mlir::failed(reduceLogicalOr.verify(loc))) {
        return mlir::failure();
    }
    if (reduceLogicalOr.getAxes() != nullptr && reduceLogicalOr.getAxesValue().has_value()) {
        return errorAt(loc, "Ambiguous axes representation");
    } else if (reduceLogicalOr.getAxes() == nullptr && !reduceLogicalOr.getAxesValue().has_value()) {
        return errorAt(loc, "Axes was not provided properly");
    }

    const auto input = reduceLogicalOr.getInput();
    const auto keepDims = reduceLogicalOr.getKeepDims();

    auto axesValue = IE::extractAxes(loc, reduceLogicalOr);

    return IE::inferReduceReturnTypeComponents(loc, input, keepDims, axesValue, inferredReturnShapes);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ReduceLogicalOrOp::fold(FoldAdaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    return nullptr;
}

void vpux::IE::ReduceLogicalOrOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                              mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr<vpux::IE::ReduceLogicalOrOp>>(context);
}
