//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::IE::CopyOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::CopyOpAdaptor copyOp(operands, attrs);
    if (mlir::failed(copyOp.verify(loc))) {
        return mlir::failure();
    }

    const auto ndInType = copyOp.getInput().getType().dyn_cast<vpux::NDTypeInterface>();
    if (ndInType == nullptr) {
        return errorAt(loc, "IE::CopyOp operand must have vpux::NDTypeInterface type");
    }

    IndexedSymbolAttr outMemSpace = nullptr;
    if (copyOp.getOutMemSpace().has_value()) {
        outMemSpace = copyOp.getOutMemSpace().value();
    }
    const auto outType = ndInType.changeMemSpace(outMemSpace).cast<mlir::RankedTensorType>();

    inferredReturnShapes.emplace_back(outType.getShape(), outType.getElementType(), outType.getEncoding());
    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::CopyOp::fold(FoldAdaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    return nullptr;
}

//
// FuseCopies
//

namespace {

class FuseCopies final : public mlir::OpRewritePattern<IE::CopyOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(IE::CopyOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseCopies::matchAndRewrite(IE::CopyOp origOp, mlir::PatternRewriter& rewriter) const {
    auto producerCopyOp = origOp.getInput().getDefiningOp<IE::CopyOp>();
    if (producerCopyOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::CopyOp>(origOp, producerCopyOp.getInput(), origOp.getOutMemSpaceAttr());
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::CopyOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<FuseCopies>(ctx);
}
