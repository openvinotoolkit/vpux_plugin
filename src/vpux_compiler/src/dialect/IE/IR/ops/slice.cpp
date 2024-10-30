//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// build
//

void vpux::IE::SliceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                              ShapeRef static_offsets, ShapeRef static_sizes) {
    build(builder, state, input, static_offsets.raw(), static_sizes.raw());
}

void vpux::IE::SliceOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                              ArrayRef<int64_t> static_offsets, ArrayRef<int64_t> static_sizes) {
    build(builder, state, input, getIntArrayAttr(builder.getContext(), static_offsets),
          getIntArrayAttr(builder.getContext(), static_sizes));
}

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::IE::SliceOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::SliceOpAdaptor sliceOp(operands, attrs, prop);
    if (mlir::failed(sliceOp.verify(loc))) {
        return mlir::failure();
    }

    const auto origType = sliceOp.getSource().getType().dyn_cast<vpux::NDTypeInterface>();
    if (origType == nullptr) {
        return errorAt(loc, "IE::SliceOp operand must have vpux::NDTypeInterface type");
    }

    const auto sliceShape = parseIntArrayAttr<int64_t>(sliceOp.getStaticSizes());
    const auto sliceOffsets = parseIntArrayAttr<int64_t>(sliceOp.getStaticOffsets());

    if (sliceShape.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Slice shape '{0}' doesn't match RankedTensor rank '{1}'", sliceShape, origType.getRank());
    }
    if (sliceOffsets.size() != checked_cast<size_t>(origType.getRank())) {
        return errorAt(loc, "Slice offsets '{0}' doesn't match RankedTensor rank '{1}'", sliceOffsets,
                       origType.getRank());
    }

    const auto newType = origType.extractDenseTile(ShapeRef(sliceOffsets), ShapeRef(sliceShape));
    const auto newTensorType = newType.cast<mlir::RankedTensorType>();
    inferredReturnShapes.emplace_back(newTensorType.getShape(), newTensorType.getElementType(),
                                      newTensorType.getEncoding());

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::SliceOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (getSource().getType() == getResult().getType()) {
        return getSource();
    }

    if (const auto origContent = operands[0].dyn_cast_or_null<Const::EphemeralContentAttr>()) {
        const auto offset = Shape(parseIntArrayAttr<int64_t>(getStaticOffsets()));
        const auto shape = Shape(parseIntArrayAttr<int64_t>(getStaticSizes()));
        return static_cast<Const::ContentAttr>(origContent).transform().subview(offset, shape).get();
    }

    return nullptr;
}

namespace {

//
// ComposeSlice
//

class ComposeSlice final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(IE::SliceOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ComposeSlice::matchAndRewrite(IE::SliceOp origOp, mlir::PatternRewriter& rewriter) const {
    auto producerSliceOp = origOp.getSource().getDefiningOp<IE::SliceOp>();
    if (producerSliceOp == nullptr) {
        return mlir::failure();
    }

    auto finalOffsets = parseIntArrayAttr<int64_t>(producerSliceOp.getStaticOffsets());
    const auto secondOffsets = parseIntArrayAttr<int64_t>(origOp.getStaticOffsets());
    for (auto i : irange(finalOffsets.size())) {
        finalOffsets[i] += secondOffsets[i];
    }

    const auto finalOffsetsAttr = getIntArrayAttr(getContext(), finalOffsets);
    const auto finalShapeAttr = origOp.getStaticSizes();
    rewriter.replaceOpWithNewOp<IE::SliceOp>(origOp, producerSliceOp.getSource(), finalOffsetsAttr, finalShapeAttr);

    return mlir::success();
}

//
// ProcessNegativeOffset
//

class ProcessNegativeOffset final : public mlir::OpRewritePattern<IE::SliceOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(IE::SliceOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ProcessNegativeOffset::matchAndRewrite(IE::SliceOp origOp,
                                                           mlir::PatternRewriter& /*rewriter*/) const {
    bool negFlag = false;
    auto offsets = parseIntArrayAttr<int64_t>(origOp.getStaticOffsets());
    for (size_t i = 0; i < offsets.size(); ++i) {
        if (offsets[i] < 0) {
            negFlag = true;
            offsets[i] += getShape(origOp.getSource())[Dim(i)];
        }
    }

    if (!negFlag) {
        return mlir::failure();
    }

    const auto newOffsetAttr = getIntArrayAttr(getContext(), offsets);
    origOp.setStaticOffsetsAttr(newOffsetAttr);

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::SliceOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<ProcessNegativeOffset>(ctx);
    results.add<ComposeSlice>(ctx);
}
