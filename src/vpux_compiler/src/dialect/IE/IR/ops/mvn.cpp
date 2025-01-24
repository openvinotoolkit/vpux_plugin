//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/hw_settings.hpp"

using namespace vpux;
mlir::LogicalResult vpux::IE::MVNOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::MVNOpAdaptor mvn(operands, attrs, prop);
    if (mlir::failed(mvn.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mvn.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape();
    if (inShape.size() != 4 && inShape.size() != 5) {
        return errorAt(loc, "First input tensor should have 4 or 5 dimensions");
    }

    const auto outDesc = vpux::getTensorAttr(ctx, inType.getDimsOrder(), inType.getMemSpace());
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType(), outDesc);

    return mlir::success();
}

//
// build
//

void vpux::IE::MVNOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input,
                            ::mlir::BoolAttr across_channels, ::mlir::BoolAttr normalize_variance,
                            ::mlir::FloatAttr eps) {
    build(builder, state, input.getType(), input, across_channels, normalize_variance, eps, {});
}

//
// LegalizeEpsAttr
//

namespace {
class LegalizeEpsAttr final : public mlir::OpRewritePattern<IE::MVNOp> {
public:
    using mlir::OpRewritePattern<IE::MVNOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::MVNOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult LegalizeEpsAttr::matchAndRewrite(IE::MVNOp origOp, mlir::PatternRewriter& rewriter) const {
    auto epsAttr = origOp.getEpsAttr();
    if (epsAttr == nullptr) {
        return mlir::failure();
    }

    auto eps = epsAttr.getValueAsDouble();
    auto floatEps = checked_cast<double>(std::numeric_limits<float>::epsilon());
    if (eps >= floatEps) {
        return mlir::failure();
    }

    // Convert double epsilon or smaller value to float epsilon since MVN kernel regards it as float datatype
    const auto newEpsAttr = getFPAttr(rewriter.getContext(), floatEps);
    rewriter.replaceOpWithNewOp<IE::MVNOp>(origOp, origOp.getInput(), origOp.getAcrossChannelsAttr(),
                                           origOp.getNormalizeVarianceAttr(), newEpsAttr);
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::MVNOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<LegalizeEpsAttr>(ctx);
}
