//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::PReluOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::PReluOpAdaptor prelu(operands, attrs, prop);
    if (mlir::failed(prelu.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = prelu.getInput().getType().cast<mlir::ShapedType>();
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

mlir::LogicalResult vpux::IE::LeakyReluOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::LeakyReluOpAdaptor leaky_relu(operands, attrs, prop);
    if (mlir::failed(leaky_relu.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = leaky_relu.getInput().getType().cast<mlir::ShapedType>();
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

namespace {

class UseLeakyRelu final : public mlir::OpRewritePattern<IE::PReluOp> {
public:
    using mlir::OpRewritePattern<IE::PReluOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::PReluOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult UseLeakyRelu::matchAndRewrite(IE::PReluOp origOp, mlir::PatternRewriter& rewriter) const {
    auto negativeSlopeOp = origOp.getNegativeSlope().getDefiningOp<Const::DeclareOp>();
    if (negativeSlopeOp == nullptr || !negativeSlopeOp.getContentAttr().isSplat()) {
        return mlir::failure();
    }

    const auto negativeSlopeContent = negativeSlopeOp.getContent();
    rewriter.replaceOpWithNewOp<IE::LeakyReluOp>(origOp, origOp.getType(), origOp.getInput(),
                                                 rewriter.getF64FloatAttr(negativeSlopeContent.getSplatValue<float>()));

    return mlir::success();
}

class legalizeslope final : public mlir::OpRewritePattern<IE::PReluOp> {
public:
    using mlir::OpRewritePattern<IE::PReluOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::PReluOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult legalizeslope::matchAndRewrite(IE::PReluOp origOp, mlir::PatternRewriter& rewriter) const {
    auto input = origOp.getInput();
    auto negativeSlopeOp = origOp.getNegativeSlope();

    if (negativeSlopeOp == nullptr) {
        return mlir::failure();
    }

    auto inputShape = getShape(input);
    auto slopeShape = getShape(negativeSlopeOp);

    if (slopeShape.size() == inputShape.size() &&
        (slopeShape.totalSize() == 1 || slopeShape[Dims4D::Act::C] == inputShape[Dims4D::Act::C])) {
        return mlir::failure();
    }

    SmallVector<int64_t> newShape(inputShape.size(), 1);
    newShape[Dims4D::Act::C.ind()] = slopeShape.totalSize();

    const auto newShapeAttr = getIntArrayAttr(getContext(), newShape);
    auto slopeReshape = rewriter.createOrFold<IE::ReshapeOp>(origOp->getLoc(), origOp.getNegativeSlope(), nullptr,
                                                             false, newShapeAttr);
    rewriter.replaceOpWithNewOp<IE::PReluOp>(origOp, origOp.getInput(), slopeReshape);

    return mlir::success();
}

}  // namespace

void vpux::IE::PReluOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<legalizeslope>(context);
    patterns.add<UseLeakyRelu>(context);
}
