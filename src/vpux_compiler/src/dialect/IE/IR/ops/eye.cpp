//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/attributes_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::EyeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::EyeOpAdaptor eye(operands, attrs, prop);
    if (mlir::failed(eye.verify(loc))) {
        return mlir::failure();
    }

    const auto numRows = getConstOrAttrValue(eye.getNumRows(), eye.getNumRowsValueAttr());
    if (mlir::failed(numRows)) {
        return mlir::failure();
    }

    const auto numColumns = getConstOrAttrValue(eye.getNumColumns(), eye.getNumColumnsValueAttr());
    if (mlir::failed(numColumns)) {
        return mlir::failure();
    }

    SmallVector<int64_t> batchShapeVal = {0};
    if (eye.getBatchShape() != nullptr || eye.getBatchShapeValue().has_value()) {
        auto batchShape = getConstOrArrAttrValue(eye.getBatchShape(), eye.getBatchShapeValueAttr());
        if (mlir::failed(batchShape)) {
            return mlir::failure();
        }
        batchShapeVal = batchShape.value();
    }

    const auto numRowsVal = numRows.value();
    const auto numColumnsVal = numColumns.value();
    SmallVector<int64_t> outShape = {numRowsVal, numColumnsVal};

    if (batchShapeVal[0] != 0) {
        for (size_t i = 0; i < batchShapeVal.size(); i++) {
            outShape.insert(outShape.begin() + i, batchShapeVal[i]);
        }
    }

    inferredReturnShapes.emplace_back(outShape, eye.getOutputType());

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::EyeOp> {
public:
    using mlir::OpRewritePattern<IE::EyeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::EyeOp eyeOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::EyeOp eyeOp, mlir::PatternRewriter& rewriter) const {
    if (eyeOp.getNumRowsValue().has_value() || eyeOp.getNumColumnsValue().has_value() ||
        eyeOp.getBatchShapeValue().has_value()) {
        return mlir::failure();
    }

    const auto numRows = getConstValue(eyeOp.getNumRows());
    if (mlir::failed(numRows)) {
        return mlir::failure();
    }

    const auto numColumns = getConstValue(eyeOp.getNumColumns());
    if (mlir::failed(numColumns)) {
        return mlir::failure();
    }

    SmallVector<int64_t> batchShapeVal = {0};
    if (eyeOp.getBatchShape() != nullptr) {
        const auto batchShape = getConstArrValue(eyeOp.getBatchShape());
        if (mlir::failed(batchShape)) {
            return mlir::failure();
        }
        batchShapeVal = batchShape.value();
    }

    rewriter.replaceOpWithNewOp<IE::EyeOp>(
            eyeOp, nullptr, nullptr, eyeOp.getDiagonalIndex(), nullptr,
            getIntAttr(rewriter.getContext(), numRows.value()), getIntAttr(rewriter.getContext(), numColumns.value()),
            getIntArrayAttr(rewriter.getContext(), batchShapeVal), eyeOp.getOutputTypeAttr());

    return mlir::success();
}

}  // namespace

void vpux::IE::EyeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
