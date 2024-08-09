//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::SwishOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::SwishOpAdaptor swish(operands, attrs, prop);
    if (mlir::failed(swish.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = swish.getInput().getType().cast<mlir::ShapedType>();

    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::SwishOp> {
public:
    using mlir::OpRewritePattern<IE::SwishOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::SwishOp swishOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::SwishOp swishOp, mlir::PatternRewriter& rewriter) const {
    if (swishOp.getBetaValue()) {
        return mlir::failure();
    }

    float betaValue = 1.0;

    if (auto beta = swishOp.getBeta()) {
        auto betaOp = beta.getDefiningOp<Const::DeclareOp>();
        if (betaOp == nullptr || !betaOp.getContentAttr().isSplat()) {
            return mlir::failure();
        }

        const auto betaContent = betaOp.getContent();
        betaValue = betaContent.getSplatValue<float>();
    }

    rewriter.replaceOpWithNewOp<IE::SwishOp>(swishOp, swishOp.getType(), swishOp.getInput(), nullptr,
                                             rewriter.getF64FloatAttr(betaValue));

    return mlir::success();
}

}  // namespace

void vpux::IE::SwishOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
