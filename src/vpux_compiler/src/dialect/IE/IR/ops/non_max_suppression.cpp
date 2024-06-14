//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"

using namespace vpux;

namespace {

int64_t extractMaxOutputBoxesPerClass(IE::NonMaxSuppressionOpAdaptor nms) {
    int64_t maxOutputBoxesPerClass = 0;  // default value

    if (nms.getMaxOutputBoxesPerClass() != nullptr) {
        auto maxBoxesConst = nms.getMaxOutputBoxesPerClass().getDefiningOp<Const::DeclareOp>();
        if (maxBoxesConst != nullptr) {
            const auto maxBoxesContent = maxBoxesConst.getContent();
            if (maxBoxesContent.isSplat()) {
                return maxBoxesContent.getSplatValue<int64_t>();
            }
        }
    }
    if (nms.getMaxOutputBoxesPerClassValueAttr() != nullptr) {
        return nms.getMaxOutputBoxesPerClassValueAttr().getValue().getSExtValue();
    }

    return maxOutputBoxesPerClass;
}

double extractNMSAttrValue(mlir::Value constName, mlir::FloatAttr attrName) {
    double attrValue = 0.0f;
    if (constName != nullptr) {
        vpux::Const::DeclareOp attrConst = constName.getDefiningOp<Const::DeclareOp>();
        if (attrConst != nullptr) {
            vpux::Const::Content attrContent = attrConst.getContent();
            if (attrContent.isSplat()) {
                attrValue = attrContent.getSplatValue<float>();
            }
        }
    } else if (attrName != nullptr) {
        attrValue = attrName.getValueAsDouble();
    }
    return attrValue;
}

}  // namespace

mlir::LogicalResult vpux::IE::NonMaxSuppressionOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::NonMaxSuppressionOpAdaptor nms(operands, attrs);
    if (mlir::failed(nms.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = nms.getInBoxScores().getType().cast<mlir::ShapedType>();
    mlir::Type outType = mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);

    const auto maxOutputBoxesPerClass = extractMaxOutputBoxesPerClass(nms);

    const auto numBatches = inType.getShape()[0];
    const auto numClasses = inType.getShape()[1];
    const auto numBoxes = inType.getShape()[2];
    const auto minBoxes = std::min(numBoxes, maxOutputBoxesPerClass);
    const SmallVector<int64_t> outShape{minBoxes * numBatches * numClasses, 3};
    const SmallVector<int64_t> validOutputsShape{1};
    inferredReturnShapes.emplace_back(outShape, outType);
    inferredReturnShapes.emplace_back(outShape, inType.getElementType());
    inferredReturnShapes.emplace_back(validOutputsShape, outType);

    return mlir::success();
}

namespace {

//
// ConvertConstToAttr
//

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::NonMaxSuppressionOp> {
public:
    using mlir::OpRewritePattern<IE::NonMaxSuppressionOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::NonMaxSuppressionOp nmsOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::NonMaxSuppressionOp nmsOp,
                                                        mlir::PatternRewriter& rewriter) const {
    if (nmsOp.getMaxOutputBoxesPerClassValue().has_value() && nmsOp.getIouThresholdValue().has_value() &&
        nmsOp.getScoreThresholdValue().has_value() && nmsOp.getSoftNmsSigmaValue().has_value()) {
        return mlir::failure();
    }

    int64_t maxBoxesPerClassValue = extractMaxOutputBoxesPerClass(nmsOp);

    double iouThresholdValue = extractNMSAttrValue(nmsOp.getIouThreshold(), nmsOp.getIouThresholdValueAttr());

    double scoreThresholdValue = extractNMSAttrValue(nmsOp.getScoreThreshold(), nmsOp.getScoreThresholdValueAttr());

    double softNMSSigmaValue = extractNMSAttrValue(nmsOp.getSoftNmsSigma(), nmsOp.getSoftNmsSigmaValueAttr());

    rewriter.replaceOpWithNewOp<IE::NonMaxSuppressionOp>(
            nmsOp, nmsOp.getInBoxCoords(), nmsOp.getInBoxScores(), nullptr, nullptr, nullptr, nullptr,
            nmsOp.getBoxEncoding(), nmsOp.getSortResultDescending(), rewriter.getI64IntegerAttr(maxBoxesPerClassValue),
            rewriter.getF64FloatAttr(iouThresholdValue), rewriter.getF64FloatAttr(scoreThresholdValue),
            rewriter.getF64FloatAttr(softNMSSigmaValue));

    return mlir::success();
}

}  // namespace

void vpux::IE::NonMaxSuppressionOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                                mlir::MLIRContext* context) {
    patterns.insert<ConvertConstToAttr>(context);
}
