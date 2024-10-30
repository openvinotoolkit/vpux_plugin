//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::IE::ScatterElementsUpdateOp::verify() {
    if (getAxis() != nullptr) {
        auto axisNumElements = getAxis().getType().cast<vpux::NDTypeInterface>().getNumElements();
        if (axisNumElements != 1) {
            return errorAt(*this, "Axis should have only 1 element, while it has {0}", axisNumElements);
        }

        if (getAxisValue().has_value()) {
            return errorAt(*this, "Ambiguous axis representation");
        }
    }

    if (getAxis() == nullptr && !getAxisValue().has_value()) {
        return errorAt(*this, "Axis was not provided");
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::ScatterElementsUpdateOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ScatterElementsUpdateOpAdaptor scatterElementsUpdate(operands, attrs, prop);
    if (mlir::failed(scatterElementsUpdate.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = scatterElementsUpdate.getInput().getType().cast<mlir::ShapedType>();

    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::ScatterElementsUpdateOp> {
public:
    using mlir::OpRewritePattern<IE::ScatterElementsUpdateOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ScatterElementsUpdateOp scatterElementsUpdateOp,
                                        mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::ScatterElementsUpdateOp scatterElementsUpdateOp,
                                                        mlir::PatternRewriter& rewriter) const {
    auto axis = scatterElementsUpdateOp.getAxis();
    if (axis == nullptr) {
        return mlir::failure();
    }

    auto axisConst = axis.getDefiningOp<Const::DeclareOp>();
    if (axisConst == nullptr || !axisConst.getContentAttr().isSplat()) {
        return mlir::failure();
    }

    const auto axisContent = axisConst.getContent();
    const auto axisValue = static_cast<int64_t>(axisContent.getSplatValue<int32_t>());
    rewriter.replaceOpWithNewOp<IE::ScatterElementsUpdateOp>(
            scatterElementsUpdateOp, scatterElementsUpdateOp.getType(), scatterElementsUpdateOp.getInput(),
            scatterElementsUpdateOp.getIndices(), scatterElementsUpdateOp.getUpdates(), nullptr,
            rewriter.getI64IntegerAttr(axisValue), scatterElementsUpdateOp.getReductionAttr(),
            scatterElementsUpdateOp.getUseInitValAttr());
    return mlir::success();
}

}  // namespace

void vpux::IE::ScatterElementsUpdateOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                                    mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
