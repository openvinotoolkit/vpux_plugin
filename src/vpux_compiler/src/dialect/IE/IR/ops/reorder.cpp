//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

using namespace vpux;

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::ReorderOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ReorderOpAdaptor reorder(operands, attrs);
    if (mlir::failed(reorder.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = reorder.getInput().getType().cast<mlir::RankedTensorType>();

    const auto outDesc = vpux::getTensorAttr(reorder.getDstOrder(), /*memSpace=*/nullptr, getBounds(inType));

    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType(), outDesc);

    return mlir::success();
}

//
// Canonicalization
//

namespace {

#include <vpux/compiler/dialect/IE/reorder.hpp.inc>

}  // namespace

void vpux::IE::ReorderOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext*) {
    populateWithGenerated(patterns);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ReorderOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    if (const auto cst = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        return cst.reorder(DimsOrder::fromValue(getOutput()));
    }

    return nullptr;
}
