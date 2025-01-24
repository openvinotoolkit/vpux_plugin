//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ReverseOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ReverseOpAdaptor reverse(operands, attrs, prop);
    if (mlir::failed(reverse.verify(loc))) {
        return mlir::failure();
    }

    const auto dataType = mlir::cast<mlir::ShapedType>(reverse.getInput().getType());
    const auto dataShape = dataType.getShape();

    if (dataShape.size() < 1) {
        return errorAt(loc, "First input tensor's size should not be less than 1D. Got {0}D tensor", dataShape.size());
    }

    const auto elementType = dataType.getElementType();

    inferredReturnShapes.emplace_back(dataShape, elementType);

    return mlir::success();
}

namespace {

//
// getAxis
//

mlir::FailureOr<SmallVector<int64_t>> getAxis(IE::ReverseOpAdaptor reverse, mlir::Location loc) {
    if (reverse.getAxis() != nullptr && reverse.getAxisValue().has_value()) {
        return errorAt(loc, "Ambiguous axes representation");
    }
    if (reverse.getAxis() == nullptr && !reverse.getAxisValue().has_value()) {
        return errorAt(loc, "Missing axes value");
    }

    if (reverse.getAxisValue().has_value()) {
        return parseIntArrayAttr<int64_t>(reverse.getAxisValue().value());
    }

    auto axesConst = reverse.getAxis().getDefiningOp<Const::DeclareOp>();
    if (axesConst == nullptr) {
        return errorAt(loc, "Only constant axes are supported");
    }

    const auto axesContent = axesConst.getContent();
    auto axes = to_small_vector(axesContent.getValues<int64_t>());

    const auto inType = mlir::cast<mlir::ShapedType>(reverse.getInput().getType());
    const auto inRank = inType.getRank();

    for (auto& axis : axes) {
        if (axis < 0) {
            axis += inRank;
        }
    }
    std::sort(axes.begin(), axes.end());

    return axes;
}

//
// ConvertConstToAttr
//

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::ReverseOp> {
public:
    using mlir::OpRewritePattern<IE::ReverseOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ReverseOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::ReverseOp origOp, mlir::PatternRewriter& rewriter) const {
    if (origOp.getAxisValue().has_value()) {
        return mlir::failure();
    }

    const auto axes = ::getAxis(origOp, origOp->getLoc());
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    const auto axesAttr = getIntArrayAttr(getContext(), axes.value());
    rewriter.replaceOpWithNewOp<IE::ReverseOp>(origOp, origOp.getInput(), nullptr, axesAttr, origOp.getMode());
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::ReverseOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
