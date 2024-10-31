//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/pad_extract.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::PadOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::PadOpAdaptor pad(operands, attrs, prop);
    if (mlir::failed(pad.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = pad.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inType.getShape();

    auto padBegin = IE::extractPads(loc, pad.getPadsBegin(), pad.getPadsBeginAttr(), inputShape);
    if (mlir::failed(padBegin)) {
        return mlir::failure();
    }
    const auto padEnd = IE::extractPads(loc, pad.getPadsEnd(), pad.getPadsEndAttr(), inputShape);
    if (mlir::failed(padEnd)) {
        return mlir::failure();
    }
    if (pad.getMode() == IE::PadMode::CONSTANT && pad.getPadValue() == nullptr && !pad.getPadValueAttr().has_value()) {
        return errorAt(loc, "pad_mode is CONSTANT but pad_value hasn't provided");
    }

    const auto newType = inType.pad(ShapeRef(padBegin.value()), ShapeRef(padEnd.value()));
    const auto newTensorType = newType.cast<mlir::RankedTensorType>();
    inferredReturnShapes.emplace_back(newTensorType.getShape(), newTensorType.getElementType());

    return mlir::success();
}

namespace {

//
// ConvertConstToAttr
//

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::PadOp> {
public:
    using mlir::OpRewritePattern<IE::PadOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::PadOp padOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::PadOp padOp, mlir::PatternRewriter& rewriter) const {
    if (padOp.getPadsBeginAttr().has_value() || padOp.getPadsEndAttr().has_value() ||
        padOp.getPadValueAttr().has_value()) {
        return mlir::failure();
    }

    const auto inType = padOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inType.getShape();

    // convert pads_begin

    auto padsBegin = IE::extractPads(padOp.getLoc(), padOp.getPadsBegin(), padOp.getPadsBeginAttr(), inputShape);
    if (mlir::failed(padsBegin)) {
        return mlir::failure();
    }
    const auto padsBeginAttr = getIntArrayAttr(padOp.getContext(), padsBegin.value());

    // convert pads_end

    auto padsEnd = IE::extractPads(padOp.getLoc(), padOp.getPadsEnd(), padOp.getPadsEndAttr(), inputShape);
    if (mlir::failed(padsEnd)) {
        return mlir::failure();
    }
    const auto padsEndAttr = getIntArrayAttr(padOp.getContext(), padsEnd.value());

    // convert pad_value

    if (padOp.getPadValue() != nullptr) {
        const auto padValueType = padOp.getPadValue().getType().cast<mlir::ShapedType>();
        if (padValueType.getNumElements() != 1) {
            return errorAt(padOp.getLoc(), "'pad_value' should have only 1 element, while it has {0}",
                           padValueType.getNumElements());
        }

        auto padValueConst = padOp.getPadValue().getDefiningOp<Const::DeclareOp>();
        if (padValueConst == nullptr) {
            return errorAt(padOp.getLoc(), "Only constant input is supported for 'pad_value'");
        }

        if (const auto attr = padValueConst.getContentAttr(); !attr.isSplat()) {
            return errorAt(padOp.getLoc(), "Only splat input is supported for 'pad_value'");
        }

        const auto padValueContent = padValueConst.getContent();
        const auto padValue = padValueContent.getSplatValue<float>();
        const auto padValueAttr = getFPAttr(padOp.getContext(), padValue);

        rewriter.replaceOpWithNewOp<IE::PadOp>(padOp, padOp.getInput(), nullptr, nullptr, nullptr, padsBeginAttr,
                                               padsEndAttr, padValueAttr, padOp.getMode());
    } else {
        rewriter.replaceOpWithNewOp<IE::PadOp>(padOp, padOp.getInput(), nullptr, nullptr, nullptr, padsBeginAttr,
                                               padsEndAttr, nullptr, padOp.getMode());
    }
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::PadOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}

//
// fold
//

mlir::OpFoldResult vpux::IE::PadOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    VPUX_THROW_UNLESS(!operands.empty(), "Wrong number of operands : {0}", operands.size());

    if (const auto attr = operands[0].dyn_cast_or_null<Const::EphemeralContentAttr>()) {
        if (getMode() == IE::PadMode::CONSTANT) {
            if (getPadsBeginAttr().has_value() && getPadsEndAttr().has_value() && getPadValueAttr().has_value()) {
                if (getPadValueAttr()->convertToDouble() == 0.0) {
                    const auto padsBefore = Shape(parseIntArrayAttr<int64_t>(getPadsBeginAttr().value()));
                    const auto padsAfter = Shape(parseIntArrayAttr<int64_t>(getPadsEndAttr().value()));

                    return static_cast<Const::ContentAttr>(attr).transform().padWithZero(padsBefore, padsAfter).get();
                }
            }
        }
    }

    return nullptr;
}
