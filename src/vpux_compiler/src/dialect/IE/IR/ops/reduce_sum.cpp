//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ReduceSumOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ReduceSumOpAdaptor reduceSum(operands, attrs);
    if (mlir::failed(reduceSum.verify(loc))) {
        return mlir::failure();
    }
    if (reduceSum.getAxes() != nullptr && reduceSum.getAxesValue().has_value()) {
        return errorAt(loc, "Ambiguous axes representation");
    } else if (reduceSum.getAxes() == nullptr && !reduceSum.getAxesValue().has_value()) {
        return errorAt(loc, "Axes was not provided properly");
    }

    const auto input = reduceSum.getInput();
    const auto keepDims = reduceSum.getKeepDims();

    auto axesValue = IE::extractAxes(loc, reduceSum);

    return IE::inferReduceReturnTypeComponents(loc, input, keepDims, axesValue, inferredReturnShapes);
}

mlir::LogicalResult vpux::IE::ReduceSumOp::verify() {
    llvm::SmallVector<int64_t> axesVec;
    const auto op = getOperation();
    if (getAxes() != nullptr) {
        const auto opAxes = getAxes().getType().dyn_cast<mlir::RankedTensorType>();

        if (opAxes == nullptr) {
            return errorAt(op, "Axes is not a 'RankedTensorType', got '{0}'", opAxes);
        }

        const auto axesRank = opAxes.getRank();

        // The axes input must be a scalar or 1D tensor
        if (axesRank > 1) {
            return errorAt(
                    op, "Operation has unsupported tensor rank '{0}' for axes, it must be either a scalar or 1D tensor",
                    axesRank);
        }
        // The axes input must have integer type.
        if (!opAxes.getElementType().isa<mlir::IntegerType>()) {
            return errorAt(op, " Axes input must have integer element type but actual element type is '{0}'",
                           opAxes.getElementType());
        }

        // The axes input must contain unique elements
        axesVec = parseIntArrayAttr<int64_t>(vpux::IE::getIntArrayAttrValue(getAxes()));
    }

    if (getAxesValue().has_value()) {
        const auto opAxesValue = getAxesValue().value();

        // The axes input must contain unique elements
        axesVec = parseIntArrayAttr<int64_t>(opAxesValue);
    }

    llvm::sort(axesVec);
    bool isAllUnique = std::unique(axesVec.begin(), axesVec.end()) == axesVec.end();
    if (!isAllUnique) {
        return errorAt(op, "Axes values should be unique");
    }

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ReduceSumOp::fold(FoldAdaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    return nullptr;
}

void vpux::IE::ReduceSumOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr<vpux::IE::ReduceSumOp>>(context);
}
