//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

using namespace vpux;
using namespace IE;

mlir::LogicalResult vpux::IE::ShapeCastOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ShapeCastOpAdaptor shapeCast(operands, attrs);
    if (mlir::failed(shapeCast.verify(loc))) {
        return mlir::failure();
    }

    const auto outShape = parseIntArrayAttr<int64_t>(shapeCast.getShape());

    const auto inType = shapeCast.getSource().getType().cast<vpux::NDTypeInterface>();
    const auto outDesc = vpux::getTensorAttr(ctx, inType.getDimsOrder(), inType.getMemSpace());
    inferredReturnTypes.emplace_back(outShape, inType.getElementType(), outDesc);
    return mlir::success();
}

mlir::OpFoldResult vpux::IE::ShapeCastOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    auto inputType = getSource().getType().cast<vpux::NDTypeInterface>();
    auto outputType = getResult().getType().cast<vpux::NDTypeInterface>();
    if (getSource().getType() == getResult().getType()) {
        return getSource();
    }

    VPUX_THROW_UNLESS(!operands.empty(), "Wrong number of operands : {0}", operands.size());
    if (inputType.getElementType().dyn_cast_or_null<mlir::quant::UniformQuantizedPerAxisType>()) {
        return nullptr;
    }
    if (const auto attr = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        return attr.reshape(outputType.getShape());
    }

    return nullptr;
}

//
// FuseShapeCast
//

namespace {
class FuseShapeCast final : public mlir::OpRewritePattern<IE::ShapeCastOp> {
public:
    using mlir::OpRewritePattern<IE::ShapeCastOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ShapeCastOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseShapeCast::matchAndRewrite(IE::ShapeCastOp origOp, mlir::PatternRewriter& rewriter) const {
    auto prevOp = origOp.getSource().getDefiningOp<IE::ShapeCastOp>();
    if (prevOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::ShapeCastOp>(origOp, prevOp.getSource(), origOp.getShape());
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::ShapeCastOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<FuseShapeCast>(ctx);
}
