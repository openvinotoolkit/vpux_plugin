//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::LRNOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::LRNOpAdaptor lrn(operands, attrs, prop);
    if (mlir::failed(lrn.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = lrn.getInput().getType().cast<mlir::ShapedType>();
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

namespace {

//
// getAxes
//

mlir::FailureOr<SmallVector<int64_t>> getAxes(IE::LRNOpAdaptor LRN, mlir::Location loc) {
    if (LRN.getAxes() != nullptr && LRN.getAxesValue().has_value()) {
        return errorAt(loc, "Ambiguous axes representation");
    }
    if (LRN.getAxes() == nullptr && !LRN.getAxesValue().has_value()) {
        return errorAt(loc, "Missing axes value");
    }

    if (LRN.getAxesValue().has_value()) {
        return parseIntArrayAttr<int64_t>(LRN.getAxesValue().value());
    }

    auto axesConst = LRN.getAxes().getDefiningOp<Const::DeclareOp>();
    if (axesConst == nullptr) {
        return errorAt(loc, "Only constant axes are supported");
    }

    const auto axesContent = axesConst.getContent();
    auto axes = to_small_vector(axesContent.getValues<int64_t>());

    const auto inType = LRN.getInput().getType().cast<mlir::ShapedType>();
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

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::LRNOp> {
public:
    using mlir::OpRewritePattern<IE::LRNOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::LRNOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::LRNOp origOp, mlir::PatternRewriter& rewriter) const {
    if (origOp.getAxesValue().has_value()) {
        return mlir::failure();
    }

    const auto axes = ::getAxes(origOp, origOp->getLoc());
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    const auto axesAttr = getIntArrayAttr(getContext(), axes.value());
    rewriter.replaceOpWithNewOp<IE::LRNOp>(origOp, origOp.getInput(), nullptr, axesAttr, origOp.getAlpha(),
                                           origOp.getBeta(), origOp.getBias(), origOp.getSize());
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::LRNOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
