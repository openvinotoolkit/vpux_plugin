//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::LayoutCastOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::LayoutCastOpAdaptor overrideLayout(operands, attrs, prop);
    if (mlir::failed(overrideLayout.verify(loc))) {
        return mlir::failure();
    }

    const auto outAffineMap = overrideLayout.getDstOrder();
    const auto inType = overrideLayout.getInput().getType().cast<mlir::RankedTensorType>();
    const auto outDesc = vpux::getTensorAttr(outAffineMap, nullptr);
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType(), outDesc);

    return mlir::success();
}

//
// verify
//

mlir::LogicalResult vpux::IE::LayoutCastOp::verify() {
    const auto outAffineMap = getDstOrder();
    const auto inType = getInput().getType().cast<vpux::NDTypeInterface>();
    if (inType.getRank() != outAffineMap.getNumDims()) {
        return errorAt(*this, "Cannot apply {0} map to {1}.", outAffineMap, inType.getShape());
    }

    return mlir::success();
}

mlir::OpFoldResult vpux::IE::LayoutCastOp::fold(FoldAdaptor adaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }
    auto operands = adaptor.getOperands();
    if (const auto cst = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
        auto dstOrder = DimsOrder::fromAffineMap(getDstOrder());
        return static_cast<Const::ContentAttr>(cst).transform().layoutCast(dstOrder).get();
    }

    return nullptr;
}

//
// FuseLayoutCasts
//

namespace {
class FuseLayoutCasts final : public mlir::OpRewritePattern<IE::LayoutCastOp> {
public:
    using mlir::OpRewritePattern<IE::LayoutCastOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::LayoutCastOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseLayoutCasts::matchAndRewrite(IE::LayoutCastOp origOp, mlir::PatternRewriter& rewriter) const {
    // Transform
    // Input type1 -> IE.LayoutCast type2 -> IE.LayoutCast type3 -> Output type3
    // into
    // Input type1 -> IE.LayoutCast type3 -> Output type3
    auto producerOp = origOp.getInput().getDefiningOp<IE::LayoutCastOp>();
    if (producerOp == nullptr || !producerOp.getOutput().hasOneUse()) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::LayoutCastOp>(origOp, origOp.getOutput().getType(), producerOp.getInput(),
                                                  origOp.getDstOrderAttr());

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::LayoutCastOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<FuseLayoutCasts>(ctx);
}
