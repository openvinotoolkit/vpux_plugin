//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/conversion/impl/convert_layers_to_vpu_strategy.hpp"
#include "vpux/compiler/conversion/passes/IE2VPU/convert_layers_to_VPU.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

namespace {

//
// EmbeddingSegmentsSumRewriter
//

class EmbeddingSegmentsSumRewriter : public mlir::OpRewritePattern<IE::EmbeddingSegmentsSumOp> {
public:
    EmbeddingSegmentsSumRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::EmbeddingSegmentsSumOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::EmbeddingSegmentsSumOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EmbeddingSegmentsSumRewriter::matchAndRewrite(IE::EmbeddingSegmentsSumOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    const auto ctx = origOp->getContext();
    const auto weights = origOp.getPerSampleWeights();
    if (weights != nullptr) {
        rewriter.replaceOpWithNewOp<VPU::EmbeddingSegmentsSumOp>(
                origOp, origOp.getEmbTable(), origOp.getIndices(), origOp.getSegmentIds(), origOp.getPerSampleWeights(),
                /*indices_value=*/nullptr, /*segment_ids_value=*/nullptr, origOp.getNumSegmentsValueAttr(),
                origOp.getDefaultIndexValueAttr(), /*per_sample_weights_value=*/nullptr);
        return mlir::success();
    }

    mlir::RankedTensorType weightsTensorType;
    mlir::DenseElementsAttr baseAttr;
    const auto weightsShape = getShape(origOp.getIndices()).raw();
    const auto inType = origOp.getEmbTable().getType().cast<NDTypeInterface>();

    computeWeightForEmbeddingOp(ctx, weightsTensorType, baseAttr, weightsShape, inType);

    auto cstDeclOp =
            rewriter.create<Const::DeclareOp>(origOp.getLoc(), weightsTensorType, Const::ContentAttr::get(baseAttr));

    rewriter.replaceOpWithNewOp<VPU::EmbeddingSegmentsSumOp>(
            origOp, origOp.getEmbTable(), origOp.getIndices(), origOp.getSegmentIds(), cstDeclOp.getOutput(),
            /*indices_value=*/nullptr, /*segment_ids_value=*/nullptr, origOp.getNumSegmentsValueAttr(),
            origOp.getDefaultIndexValueAttr(), /*per_sample_weights_value=*/nullptr);
    return mlir::success();
}

//
// EmbeddingBagOffsetsSumRewriter
//

class EmbeddingBagOffsetsSumRewriter final : public mlir::OpRewritePattern<IE::EmbeddingBagOffsetsSumOp> {
public:
    EmbeddingBagOffsetsSumRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::EmbeddingBagOffsetsSumOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::EmbeddingBagOffsetsSumOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EmbeddingBagOffsetsSumRewriter::matchAndRewrite(IE::EmbeddingBagOffsetsSumOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("Found EmbeddingBagOffsetsSumOp Operation '{0}'", origOp->getLoc());

    const auto ctx = origOp->getContext();
    const auto weights = origOp.getPerSampleWeights();
    if (weights != nullptr) {
        rewriter.replaceOpWithNewOp<VPU::EmbeddingBagOffsetsSumOp>(
                origOp, origOp.getEmbTable(), origOp.getIndices(), origOp.getOffsets(), origOp.getPerSampleWeights(),
                /*indices_value=*/nullptr,
                /*offsets_value=*/nullptr, origOp.getDefaultIndexValueAttr(), /*per_sample_weights_value=*/nullptr);
        return mlir::success();
    }

    mlir::RankedTensorType weightsTensorType;
    mlir::DenseElementsAttr baseAttr;
    const auto weightsShape = getShape(origOp.getIndices()).raw();
    const auto inType = origOp.getEmbTable().getType().cast<NDTypeInterface>();

    computeWeightForEmbeddingOp(ctx, weightsTensorType, baseAttr, weightsShape, inType);

    auto cstDeclOp =
            rewriter.create<Const::DeclareOp>(origOp.getLoc(), weightsTensorType, Const::ContentAttr::get(baseAttr));

    rewriter.replaceOpWithNewOp<VPU::EmbeddingBagOffsetsSumOp>(
            origOp, origOp.getEmbTable(), origOp.getIndices(), origOp.getOffsets(), cstDeclOp.getOutput(),
            /*indices_value=*/nullptr, /*offsets_value=*/nullptr, origOp.getDefaultIndexValueAttr(),
            /*per_sample_weights_value=*/nullptr);

    return mlir::success();
}

//
// AccumulateRewrite
//

class AccumulateRewrite final : public mlir::OpRewritePattern<IE::AccumulateOp> {
public:
    AccumulateRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AccumulateOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AccumulateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult AccumulateRewrite::matchAndRewrite(IE::AccumulateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Found AccumulateOp Operation '{0}'", origOp->getLoc());

    if (origOp.getLhsScale() != nullptr && origOp.getRhsScale() != nullptr) {
        rewriter.replaceOpWithNewOp<VPU::AccumulateOp>(origOp, origOp.getLhs(), origOp.getRhs(), origOp.getLhsScale(),
                                                       origOp.getRhsScale(),
                                                       /*multiClusterStrategy=*/nullptr);
    } else if (origOp.getLhsScale() == nullptr && origOp.getRhsScale() == nullptr) {
        const auto broadcast = IE::AutoBroadcastType::NONE_OR_EXPLICIT;
        const auto broadcastAttr = IE::AutoBroadcastTypeAttr::get(rewriter.getContext(), broadcast);
        rewriter.replaceOpWithNewOp<VPU::AddOp>(origOp, origOp.getLhs(), origOp.getRhs(), broadcastAttr,
                                                /*postOp=*/nullptr);
    } else {
        VPUX_THROW("IE.Accumulate must set either both scales or none.");
    }

    return mlir::success();
}

}  // namespace

namespace vpux::arch37xx {
void ConvertLayers2VPUStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();
    patterns.add<EmbeddingSegmentsSumRewriter>(ctx, log);
    patterns.add<EmbeddingBagOffsetsSumRewriter>(ctx, log);
    patterns.add<AccumulateRewrite>(ctx, log);
}

}  // namespace vpux::arch37xx
