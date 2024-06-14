//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::IE::EmbeddingSegmentsSumOp::verify() {
    int64_t numElements = 0;
    const auto checkNumElements = [&](mlir::Value tensor) {
        if (tensor == nullptr) {
            return true;
        }

        numElements = tensor.getType().cast<vpux::NDTypeInterface>().getNumElements();
        return numElements == 1;
    };

    if (!checkNumElements(getNumSegments())) {
        return errorAt(*this, "num_segments should have only 1 element, while it has {0}", numElements);
    }

    if (!checkNumElements(getDefaultIndex())) {
        return errorAt(*this, "default_index should have only 1 element, while it has {0}", numElements);
    }

    return mlir::success();
}

namespace {

mlir::FailureOr<int64_t> extractNumSegments(mlir::Location loc,
                                            IE::EmbeddingSegmentsSumOpAdaptor embeddingSegmentsSum) {
    if (embeddingSegmentsSum.getNumSegments() != nullptr) {
        auto numSegmentsConst = embeddingSegmentsSum.getNumSegments().getDefiningOp<Const::DeclareOp>();
        if (numSegmentsConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for numSegments");
        }

        const auto numSegmentsContent = numSegmentsConst.getContent();
        if (!numSegmentsContent.isSplat()) {
            return errorAt(loc, "numSegments value must be a scalar");
        }

        int64_t numSegments = numSegmentsContent.getSplatValue<int64_t>();
        return numSegments;
    } else if (embeddingSegmentsSum.getNumSegmentsValue().has_value()) {
        return embeddingSegmentsSum.getNumSegmentsValue().value();
    }
    return errorAt(loc, "NumSegments was not provided");
}

}  // namespace

mlir::LogicalResult vpux::IE::EmbeddingSegmentsSumOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::EmbeddingSegmentsSumOpAdaptor embeddingSegmentsSum(operands, attrs);
    if (mlir::failed(embeddingSegmentsSum.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = embeddingSegmentsSum.getEmbTable().getType().cast<mlir::ShapedType>();

    const auto numSegments = extractNumSegments(loc, embeddingSegmentsSum);
    if (mlir::failed(numSegments)) {
        return mlir::failure();
    }

    int64_t numSegmentsVal = checked_cast<int64_t>(*numSegments);

    SmallVector<int64_t> outShape(to_small_vector(inType.getShape()));
    outShape[0] = numSegmentsVal;

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}

//
// ConvertConstToAttrVPUX30XX
//

namespace {
class ConvertConstToAttrVPUX30XX final : public mlir::OpRewritePattern<IE::EmbeddingSegmentsSumOp> {
public:
    using mlir::OpRewritePattern<IE::EmbeddingSegmentsSumOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::EmbeddingSegmentsSumOp EmbeddingSegmentsSumOp,
                                        mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttrVPUX30XX::matchAndRewrite(IE::EmbeddingSegmentsSumOp embeddingSegmentsSumOp,
                                                                mlir::PatternRewriter& rewriter) const {
    const auto module = embeddingSegmentsSumOp->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);
    if (arch != VPU::ArchKind::NPU30XX) {
        return mlir::failure();
    }
    auto indicesAttr = vpux::IE::getIntArrayAttrValue(embeddingSegmentsSumOp.getIndices());
    auto segmentIdsAttr = vpux::IE::getIntArrayAttrValue(embeddingSegmentsSumOp.getSegmentIds());
    auto numSegmentsAttr = vpux::IE::getIntAttrValue(embeddingSegmentsSumOp.getNumSegments(), rewriter);
    auto defaultIndexAttr = vpux::IE::getIntAttrValue(embeddingSegmentsSumOp.getDefaultIndex(), rewriter);
    auto perSampleWeightsAttr = vpux::IE::getFloatArrayAttrValue(embeddingSegmentsSumOp.getPerSampleWeights());

    if ((embeddingSegmentsSumOp.getDefaultIndexValueAttr() == nullptr) && (defaultIndexAttr == nullptr)) {
        int32_t defaultValueDefaultIndex =
                -1;  // The OpenVINO spec expects default value 0. However, the ACT Shave kernel implementation fills
                     // the empty segments with zero when a negative value is provided.
        defaultIndexAttr = rewriter.getI32IntegerAttr(defaultValueDefaultIndex);
    }

    if ((embeddingSegmentsSumOp.getPerSampleWeightsValueAttr() == nullptr) && (perSampleWeightsAttr == nullptr)) {
        SmallVector<float> defaultValuePerSampleWeights(indicesAttr.size(), 1);
        perSampleWeightsAttr = getFPArrayAttr(embeddingSegmentsSumOp.getContext(), defaultValuePerSampleWeights);
    }

    if ((indicesAttr == nullptr) && (segmentIdsAttr == nullptr) && (numSegmentsAttr == nullptr) &&
        (defaultIndexAttr == nullptr) && (perSampleWeightsAttr == nullptr)) {
        return mlir::failure();
    }

    const auto indices = (indicesAttr == nullptr) ? embeddingSegmentsSumOp.getIndices()
                                                  : mlir::TypedValue<mlir::RankedTensorType>(nullptr);
    const auto segmentIds = (segmentIdsAttr == nullptr) ? embeddingSegmentsSumOp.getSegmentIds()
                                                        : mlir::TypedValue<mlir::RankedTensorType>(nullptr);
    const auto numSegments = (numSegmentsAttr == nullptr) ? embeddingSegmentsSumOp.getNumSegments()
                                                          : mlir::TypedValue<mlir::RankedTensorType>(nullptr);
    const auto defaultIndex = (defaultIndexAttr == nullptr) ? embeddingSegmentsSumOp.getDefaultIndex()
                                                            : mlir::TypedValue<mlir::RankedTensorType>(nullptr);
    const auto perSampleWeights = (perSampleWeightsAttr == nullptr) ? embeddingSegmentsSumOp.getPerSampleWeights()
                                                                    : mlir::TypedValue<mlir::RankedTensorType>(nullptr);

    rewriter.replaceOpWithNewOp<IE::EmbeddingSegmentsSumOp>(
            embeddingSegmentsSumOp, embeddingSegmentsSumOp.getType(), embeddingSegmentsSumOp.getEmbTable(), indices,
            segmentIds, numSegments, defaultIndex, perSampleWeights, indicesAttr, segmentIdsAttr, numSegmentsAttr,
            defaultIndexAttr, perSampleWeightsAttr);

    return mlir::success();
}

}  // namespace

//
// ConvertConstToAttrVPUX37XX
//

namespace {
class ConvertConstToAttrVPUX37XX final : public mlir::OpRewritePattern<IE::EmbeddingSegmentsSumOp> {
public:
    using mlir::OpRewritePattern<IE::EmbeddingSegmentsSumOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::EmbeddingSegmentsSumOp EmbeddingSegmentsSumOp,
                                        mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttrVPUX37XX::matchAndRewrite(IE::EmbeddingSegmentsSumOp embeddingSegmentsSumOp,
                                                                mlir::PatternRewriter& rewriter) const {
    const auto module = embeddingSegmentsSumOp->getParentOfType<mlir::ModuleOp>();
    const auto arch = VPU::getArch(module);
    const std::set<VPU::ArchKind> compatibleTargets = {
            VPU::ArchKind::NPU37XX,
            VPU::ArchKind::NPU40XX,
    };
    if (compatibleTargets.count(arch) <= 0) {
        return mlir::failure();
    }
    auto numSegmentsAttr = vpux::IE::getIntAttrValue(embeddingSegmentsSumOp.getNumSegments(), rewriter);
    auto defaultIndexAttr = vpux::IE::getIntAttrValue(embeddingSegmentsSumOp.getDefaultIndex(), rewriter);

    if ((embeddingSegmentsSumOp.getNumSegments() == nullptr) && (embeddingSegmentsSumOp.getDefaultIndex() == nullptr)) {
        return mlir::failure();
    }

    if (defaultIndexAttr == nullptr) {
        int32_t defaultValueDefaultIndex = -1;
        defaultIndexAttr = rewriter.getI32IntegerAttr(defaultValueDefaultIndex);
    }

    rewriter.replaceOpWithNewOp<IE::EmbeddingSegmentsSumOp>(
            embeddingSegmentsSumOp, embeddingSegmentsSumOp.getType(), embeddingSegmentsSumOp.getEmbTable(),
            embeddingSegmentsSumOp.getIndices(), embeddingSegmentsSumOp.getSegmentIds(), nullptr, nullptr,
            embeddingSegmentsSumOp.getPerSampleWeights(), nullptr, nullptr, numSegmentsAttr, defaultIndexAttr, nullptr);
    return mlir::success();
}

}  // namespace

void vpux::IE::EmbeddingSegmentsSumOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                                   mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttrVPUX30XX>(context);
    patterns.add<ConvertConstToAttrVPUX37XX>(context);
}
