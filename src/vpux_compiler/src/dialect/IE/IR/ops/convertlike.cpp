//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ConvertLikeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ConvertLikeOpAdaptor cvt(operands, attrs);
    if (mlir::failed(cvt.verify(loc))) {
        return mlir::failure();
    }
    const auto data = cvt.getInput().getType().cast<mlir::ShapedType>();
    const auto like = cvt.getLike().getType().cast<mlir::ShapedType>();

    inferredReturnShapes.emplace_back(data.getShape(), like.getElementType());
    return mlir::success();
}

//
// ConvertConvertLikeToConvert
//

class ConvertConvertLikeToConvert final : public mlir::OpRewritePattern<IE::ConvertLikeOp> {
public:
    using mlir::OpRewritePattern<IE::ConvertLikeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvertLikeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConvertLikeToConvert::matchAndRewrite(IE::ConvertLikeOp convertLikeOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    const auto outputType = convertLikeOp.getOutput().getType().cast<vpux::NDTypeInterface>().getElementType();

    rewriter.replaceOpWithNewOp<IE::ConvertOp>(convertLikeOp, convertLikeOp.getInput(), outputType);

    return mlir::success();
}

void vpux::IE::ConvertLikeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                          mlir::MLIRContext* context) {
    patterns.add<ConvertConvertLikeToConvert>(context);
}

mlir::OpFoldResult vpux::IE::ConvertLikeOp::fold(FoldAdaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    return nullptr;
}
