//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"

using namespace vpux;

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::PermuteQuantizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::PermuteQuantizeOpAdaptor permute_quantize(operands, attrs, prop);
    if (mlir::failed(permute_quantize.verify(loc))) {
        return mlir::failure();
    }

    mlir::Value input = permute_quantize.getInput();
    mlir::AffineMap memPerm = permute_quantize.getMemPerm();
    mlir::AffineMap dstOrder = permute_quantize.getDstOrder();
    const auto dstElemType = permute_quantize.getDstElemType();

    const auto padBegin = parseIntArrayAttr<int64_t>(permute_quantize.getPadsBegin());
    const auto padEnd = parseIntArrayAttr<int64_t>(permute_quantize.getPadsEnd());

    const auto inType = permute_quantize.getInput().getType().cast<vpux::NDTypeInterface>();

    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder = DimsOrder::fromAffineMap(dstOrder);

    const auto newType = inType.pad(ShapeRef(padBegin), ShapeRef(padEnd));
    const auto inShapeExpanded = newType.getShape();

    const auto inMemShape = inOrder.toMemoryOrder(inShapeExpanded);
    const auto outMemShape = applyPerm(inMemShape, memPerm);
    const auto outShape = outOrder.toLogicalOrder(outMemShape);

    const auto outDesc = vpux::getTensorAttr(dstOrder, nullptr);

    inferredReturnShapes.emplace_back(outShape.raw(), dstElemType, outDesc);

    return mlir::success();
}

namespace {

//
// ConvertToPermuteCast
//

class ConvertToPermuteCast final : public mlir::OpRewritePattern<IE::PermuteQuantizeOp> {
public:
    using mlir::OpRewritePattern<IE::PermuteQuantizeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::PermuteQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertToPermuteCast::matchAndRewrite(IE::PermuteQuantizeOp origOp,
                                                          mlir::PatternRewriter& rewriter) const {
    const auto inOrder = DimsOrder::fromValue(origOp.getInput());
    const auto inShape = getShape(origOp.getInput());
    const auto inMemShape = inOrder.toMemoryOrder(inShape);

    const auto inputType = origOp.getInput().getType().cast<NDTypeInterface>().getElementType();
    const auto outputType = origOp.getOutput().getType().cast<NDTypeInterface>().getElementType();

    if (!isTrivialPermute(inMemShape, origOp.getMemPerm()) || inputType != outputType ||
        inShape != getShape(origOp.getOutput())) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::PermuteCastOp>(origOp, origOp.getInput(), origOp.getDstOrderAttr(),
                                                   origOp.getMemPermAttr());
    return mlir::success();
}

}  // namespace

void vpux::IE::PermuteQuantizeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                              mlir::MLIRContext* context) {
    patterns.add<ConvertToPermuteCast>(context);
}

mlir::OpFoldResult vpux::IE::PermuteQuantizeOp::fold(FoldAdaptor) {
    if (getInput().getType() == getOutput().getType() && getMemPerm().isIdentity()) {
        return getInput();
    }

    return nullptr;
}
