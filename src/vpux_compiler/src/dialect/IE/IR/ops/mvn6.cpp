//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;
mlir::LogicalResult vpux::IE::MVN6Op::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::MVN6OpAdaptor mvn(operands, attrs, prop);
    if (mlir::failed(mvn.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mvn.getInput().getType().cast<mlir::ShapedType>();
    const auto rankedInType = mvn.getInput().getType().cast<mlir::RankedTensorType>();
    const auto outDesc = vpux::getTensorAttr(rankedInType);
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType(), outDesc);

    return mlir::success();
}

namespace {

//
// getAxes
//

mlir::FailureOr<SmallVector<int64_t>> getAxes(IE::MVN6OpAdaptor mvn, mlir::Location loc) {
    if (mvn.getAxes() != nullptr && mvn.getAxesValue().has_value()) {
        return errorAt(loc, "Ambiguous axes representation");
    }
    if (mvn.getAxes() == nullptr && !mvn.getAxesValue().has_value()) {
        return errorAt(loc, "Missing axes value");
    }

    if (mvn.getAxesValue().has_value()) {
        return parseIntArrayAttr<int64_t>(mvn.getAxesValue().value());
    }

    auto axesConst = mvn.getAxes().getDefiningOp<Const::DeclareOp>();
    if (axesConst == nullptr) {
        return errorAt(loc, "Only constant axes are supported");
    }

    const auto axesContent = axesConst.getContent();
    auto axes = to_small_vector(axesContent.getValues<int64_t>());

    const auto inType = mvn.getInput().getType().cast<mlir::ShapedType>();
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

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::MVN6Op> {
public:
    using mlir::OpRewritePattern<IE::MVN6Op>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::MVN6Op origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::MVN6Op origOp, mlir::PatternRewriter& rewriter) const {
    if (origOp.getAxesValue().has_value()) {
        return mlir::failure();
    }

    const auto axes = ::getAxes(origOp, origOp->getLoc());
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    const auto axesAttr = getIntArrayAttr(getContext(), axes.value());
    rewriter.replaceOpWithNewOp<IE::MVN6Op>(origOp, origOp.getInput(), nullptr, axesAttr,
                                            origOp.getNormalizeVarianceAttr(), origOp.getEpsAttr(),
                                            origOp.getEpsModeAttr());
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::MVN6Op::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
