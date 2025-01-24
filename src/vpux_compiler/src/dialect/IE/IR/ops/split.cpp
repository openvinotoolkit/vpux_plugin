//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// inferReturnTypeComponents
//

namespace {

Dim normalizeAxis(IE::SplitOpAdaptor split) {
    VPUX_THROW_UNLESS(split.getAxisValue().has_value(), "Got non constant axis");

    const auto inType = split.getInput().getType().cast<mlir::ShapedType>();
    const auto inRank = inType.getRank();

    auto axisInd = split.getAxisValue().value();

    // Negative value means counting dimension from the end
    if (axisInd < 0) {
        axisInd += inRank;
    }

    VPUX_THROW_UNLESS(axisInd >= 0 && axisInd < inRank, "Got wrong Split axis '{0}', out of range '{1}'", axisInd,
                      inRank);

    return Dim(axisInd);
}

mlir::FailureOr<Dim> extractAxis(mlir::Location loc, IE::SplitOpAdaptor split) {
    if (split.getAxis() != nullptr) {
        auto axisConst = split.getAxis().getDefiningOp<Const::DeclareOp>();
        if (axisConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for axis");
        }

        if (const auto& attr = axisConst.getContentAttr(); !attr.isSplat()) {
            return errorAt(loc, "Axis value must be a scalar");
        }

        const auto axisContent = axisConst.getContent();
        const auto inType = split.getInput().getType().cast<mlir::ShapedType>();
        const auto inRank = inType.getRank();

        auto axisInd = axisContent.getSplatValue<int64_t>();

        // Negative value means counting dimension from the end
        if (axisInd < 0) {
            axisInd += inRank;
        }

        VPUX_THROW_UNLESS(axisInd >= 0 && axisInd < inRank, "Got wrong Split axis '{0}', out of range '{1}'", axisInd,
                          inRank);

        return Dim(axisInd);
    } else if (split.getAxisValue().has_value()) {
        return normalizeAxis(split);
    } else {
        return errorAt(loc, "Axis was not provided");
    }
}

}  // namespace

mlir::LogicalResult vpux::IE::SplitOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::SplitOpAdaptor split(operands, attrs, prop);
    if (mlir::failed(split.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = split.getInput().getType().cast<mlir::RankedTensorType>();

    const auto axis = extractAxis(loc, split);
    if (mlir::failed(axis)) {
        return mlir::failure();
    }

    const auto num_splits = split.getNumSplits();

    auto outShape = inType.cast<vpux::NDTypeInterface>().getShape().toValues();
    if ((outShape[*axis] < num_splits) || (outShape[*axis] % num_splits != 0)) {
        return errorAt(loc, "Unsupported num_splits parameter");
    }
    outShape[*axis] /= num_splits;

    const auto elemType = inType.getElementType();
    const auto outDesc = vpux::getTensorAttr(inType);

    for (int i = 0; i < num_splits; ++i) {
        inferredReturnShapes.emplace_back(outShape.raw(), elemType, outDesc);
    }

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::SplitOp> {
public:
    using mlir::OpRewritePattern<IE::SplitOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::SplitOp splitOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::SplitOp splitOp, mlir::PatternRewriter& rewriter) const {
    if (splitOp.getAxisValue().has_value()) {
        return mlir::failure();
    }

    const auto axis = extractAxis(splitOp.getLoc(), splitOp);
    if (mlir::failed(axis)) {
        return mlir::failure();
    }

    const auto axisAttr = getIntAttr(splitOp.getContext(), axis->ind());
    rewriter.replaceOpWithNewOp<IE::SplitOp>(splitOp, splitOp.getInput(), nullptr, splitOp.getNumSplitsAttr(),
                                             axisAttr);

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::SplitOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
