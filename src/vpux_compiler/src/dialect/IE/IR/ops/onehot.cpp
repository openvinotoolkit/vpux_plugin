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

mlir::LogicalResult vpux::IE::OneHotOp::verify() {
    int64_t numElements = 0;
    const auto checkNumElements = [&](mlir::Value tensor) {
        if (tensor == nullptr) {
            return true;
        }

        numElements = tensor.getType().cast<vpux::NDTypeInterface>().getNumElements();
        return numElements == 1;
    };

    if (!checkNumElements(getDepth())) {
        return errorAt(*this, "Depth should have only 1 element, while it has {0}", numElements);
    }

    if (!checkNumElements(getOnValue())) {
        return errorAt(*this, "on_value should have only 1 element, while it has {0}", numElements);
    }

    if (!checkNumElements(getOffValue())) {
        return errorAt(*this, "off_value should have only 1 element, while it has {0}", numElements);
    }

    return mlir::success();
}

mlir::FailureOr<int64_t> extractDepth(mlir::Location loc, const mlir::Value& depth, mlir::IntegerAttr depthAttr) {
    if (depthAttr != nullptr) {
        return depthAttr.getInt();
    } else if (depth != nullptr) {
        auto depthConst = depth.getDefiningOp<Const::DeclareOp>();
        if (depthConst == nullptr) {
            return errorAt(loc, "Only constant input is supported");
        }

        if (const auto& attr = depthConst.getContentAttr(); !attr.isSplat()) {
            return errorAt(loc, "OneHot depth must be a scalar");
        }
        const auto depthContent = depthConst.getContent();
        return depthContent.getSplatValue<int64_t>();
    }

    return errorAt(loc, "depth is not provided");
}

mlir::LogicalResult vpux::IE::OneHotOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));
    IE::OneHotOpAdaptor oneHot(operands, attrs, prop);

    if (mlir::failed(oneHot.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = oneHot.getInput().getType().cast<mlir::ShapedType>();
    const auto outElemType = oneHot.getOutputType();

    auto outShape = to_small_vector(inType.getShape());
    const auto axis = oneHot.getAxisAttr();
    auto depth = extractDepth(loc, oneHot.getDepth(), oneHot.getDepthAttrAttr());
    if (mlir::failed(depth)) {
        return mlir::failure();
    }
    int64_t depthVal = checked_cast<int64_t>(*depth);
    if (axis < 0) {
        outShape.insert(outShape.end() + 1 + axis, depthVal);
    } else {
        outShape.insert(outShape.begin() + axis, depthVal);
    }

    inferredReturnShapes.emplace_back(outShape, outElemType);

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::OneHotOp> {
public:
    using mlir::OpRewritePattern<IE::OneHotOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::OneHotOp oneHotOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::OneHotOp oneHotOp, mlir::PatternRewriter& rewriter) const {
    auto depth = oneHotOp.getDepth();
    auto onValue = oneHotOp.getOnValue();
    auto offValue = oneHotOp.getOffValue();

    if ((depth == nullptr) || (onValue == nullptr) || (offValue == nullptr)) {
        return mlir::failure();
    }

    auto depthConst = depth.getDefiningOp<Const::DeclareOp>();
    auto onValueConst = onValue.getDefiningOp<Const::DeclareOp>();
    auto offValueConst = offValue.getDefiningOp<Const::DeclareOp>();

    const auto isSplat = [](Const::DeclareOp op) {
        return (op != nullptr) && op.getContentAttr().isSplat();
    };

    if (!isSplat(depthConst) || !isSplat(onValueConst) || !isSplat(offValueConst)) {
        return mlir::failure();
    }

    const auto depthContent = depthConst.getContent();
    const auto onValueContent = onValueConst.getContent();
    const auto offValueContent = offValueConst.getContent();

    const auto depthAttrValue = depthContent.getSplatValue<int64_t>();
    const auto onValueAttrValue = onValueContent.getSplatValue<float>();
    const auto offValueAttrValue = offValueContent.getSplatValue<float>();

    rewriter.replaceOpWithNewOp<IE::OneHotOp>(
            oneHotOp, oneHotOp.getType(), oneHotOp.getInput(), nullptr, nullptr, nullptr,
            rewriter.getI64IntegerAttr(depthAttrValue), rewriter.getF64FloatAttr(onValueAttrValue),
            rewriter.getF64FloatAttr(offValueAttrValue), oneHotOp.getAxisAttr(), oneHotOp.getOutputType());

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::OneHotOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
