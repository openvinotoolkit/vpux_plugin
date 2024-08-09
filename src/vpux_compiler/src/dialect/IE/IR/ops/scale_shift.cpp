//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// add

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

using namespace vpux;

#include <mlir/IR/PatternMatch.h>

namespace {

//
// FuseScaleAndBias
//

class FuseScaleAndBias final : public mlir::OpRewritePattern<IE::ScaleShiftOp> {
public:
    using mlir::OpRewritePattern<IE::ScaleShiftOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ScaleShiftOp biasOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseScaleAndBias::matchAndRewrite(IE::ScaleShiftOp biasOp, mlir::PatternRewriter& rewriter) const {
    static const auto C = Dim(1);

    if (!biasOp.getInput().hasOneUse()) {
        return mlir::failure();
    }

    if (biasOp.getWeights() != nullptr) {
        return mlir::failure();
    }

    auto scaleOp = mlir::dyn_cast_or_null<IE::ScaleShiftOp>(biasOp.getInput().getDefiningOp());
    if (scaleOp == nullptr || scaleOp.getBiases() != nullptr) {
        return mlir::failure();
    }

    auto mulOutShape = getShape(scaleOp.getOutput());
    auto weightsShape = getShape(scaleOp.getWeights());
    auto biasShape = getShape(biasOp.getBiases());

    if (mulOutShape.size() != 4) {
        return mlir::failure();
    }
    if (biasShape[C] != weightsShape[C]) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::ScaleShiftOp>(biasOp, biasOp.getType(), scaleOp.getInput(), scaleOp.getWeights(),
                                                  biasOp.getBiases());

    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::IE::ScaleShiftOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ScaleShiftOpAdaptor scaleShift(operands, attrs, prop);
    if (mlir::failed(scaleShift.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = scaleShift.getInput().getType().cast<mlir::ShapedType>();
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

void vpux::IE::ScaleShiftOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                         mlir::MLIRContext* context) {
    patterns.add<FuseScaleAndBias>(context);
}
