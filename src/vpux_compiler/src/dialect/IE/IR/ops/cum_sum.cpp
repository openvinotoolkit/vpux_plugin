//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::IE::CumSumOp::verify() {
    if (getAxis() != nullptr) {
        auto axisNumElements = getAxis().getType().cast<vpux::NDTypeInterface>().getNumElements();
        if (axisNumElements != 1) {
            return errorAt(*this, "Axis should have only 1 element, while it has {0}", axisNumElements);
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::CumSumOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::CumSumOpAdaptor cumsum(operands, attrs);
    if (mlir::failed(cumsum.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = cumsum.getInput().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();

    inferredReturnShapes.emplace_back(inShape, inType.getElementType());

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::CumSumOp> {
public:
    using mlir::OpRewritePattern<IE::CumSumOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::CumSumOp cumsumOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::CumSumOp cumsumOp, mlir::PatternRewriter& rewriter) const {
    auto axis = cumsumOp.getAxis();
    if (axis == nullptr) {
        return mlir::failure();
    }

    auto axisConst = cumsumOp.getAxis().getDefiningOp<Const::DeclareOp>();
    if (axisConst == nullptr) {
        return mlir::failure();
    }

    const auto axisContent = axisConst.getContent();
    if (!axisContent.isSplat()) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::CumSumOp>(cumsumOp, cumsumOp.getType(), cumsumOp.getInput(), nullptr,
                                              rewriter.getI64IntegerAttr(axisContent.getSplatValue<int64_t>()),
                                              cumsumOp.getExclusiveAttr(), cumsumOp.getReverseAttr());
    return mlir::success();
}

}  // namespace

void vpux::IE::CumSumOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.insert<ConvertConstToAttr>(context);
}
