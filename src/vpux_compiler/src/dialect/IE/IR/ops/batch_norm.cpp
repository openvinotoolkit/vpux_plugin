//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::BatchNormInferenceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::BatchNormInferenceOpAdaptor norm(operands, attrs, prop);
    if (mlir::failed(norm.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = norm.getInput().getType().cast<mlir::ShapedType>();
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::BatchNormInferenceOp> {
public:
    using mlir::OpRewritePattern<IE::BatchNormInferenceOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::BatchNormInferenceOp normOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::BatchNormInferenceOp normOp,
                                                        mlir::PatternRewriter& rewriter) const {
    auto gamma = normOp.getGamma();
    auto beta = normOp.getBeta();
    auto mean = normOp.getMean();
    auto var = normOp.getVariance();

    if ((gamma == nullptr) || (beta == nullptr) || (mean == nullptr) || (var == nullptr)) {
        // already converted
        return mlir::failure();
    }

    auto gammaConst = gamma.getDefiningOp<Const::DeclareOp>();
    auto betaConst = beta.getDefiningOp<Const::DeclareOp>();
    auto meanConst = mean.getDefiningOp<Const::DeclareOp>();
    auto varConst = var.getDefiningOp<Const::DeclareOp>();

    // fail if not constants
    if ((gammaConst == nullptr) || (betaConst == nullptr) || (meanConst == nullptr) || (varConst == nullptr)) {
        return mlir::failure();
    }

    const auto gammaContent = gammaConst.getContent();
    const auto betaContent = betaConst.getContent();
    const auto meanContent = meanConst.getContent();
    const auto varContent = varConst.getContent();

    const auto gammaAttr = getFPArrayAttr(getContext(), gammaContent.getValues<double>());
    const auto betaAttr = getFPArrayAttr(getContext(), betaContent.getValues<double>());
    const auto meanAttr = getFPArrayAttr(getContext(), meanContent.getValues<double>());
    const auto varAttr = getFPArrayAttr(getContext(), varContent.getValues<double>());

    rewriter.replaceOpWithNewOp<IE::BatchNormInferenceOp>(normOp, normOp.getType(), normOp.getInput(), nullptr, nullptr,
                                                          nullptr, nullptr, gammaAttr, betaAttr, meanAttr, varAttr,
                                                          normOp.getEps());
    return mlir::success();
}

}  // namespace

void vpux::IE::BatchNormInferenceOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                                 mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
