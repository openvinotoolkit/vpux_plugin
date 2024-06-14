//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/VPU30XX/conversion/impl/convert_layers_to_vpu_strategy.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

// //
// // Generated
// //

// #include <vpux/compiler/conversion/convert_layers_to_VPU.hpp.inc>

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
    rewriter.replaceOpWithNewOp<VPU::EmbeddingSegmentsSumOp>(
            origOp, origOp.getEmbTable(), /*indices=*/nullptr, /*segment_ids=*/nullptr,
            /*per_sample_weights=*/nullptr, origOp.getIndicesValueAttr(), origOp.getSegmentIdsValueAttr(),
            origOp.getNumSegmentsValueAttr(), origOp.getDefaultIndexValueAttr(), origOp.getPerSampleWeightsValueAttr());
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
    rewriter.replaceOpWithNewOp<VPU::EmbeddingBagOffsetsSumOp>(
            origOp, origOp.getEmbTable(), /*indices=*/nullptr, /*offsets=*/nullptr, /*per_sample_weights=*/nullptr,
            origOp.getIndicesValueAttr(), origOp.getOffsetsValueAttr(), origOp.getDefaultIndexValueAttr(),
            origOp.getPerSampleWeightsValueAttr());
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

    VPUX_THROW_UNLESS(origOp.getLhsScale() == nullptr && origOp.getRhsScale() == nullptr,
                      "IE.Accumulate does not support scaling for this target.");
    const auto broadcast = IE::AutoBroadcastType::NONE_OR_EXPLICIT;
    const auto broadcastAttr = IE::AutoBroadcastTypeAttr::get(rewriter.getContext(), broadcast);
    rewriter.replaceOpWithNewOp<VPU::AddOp>(origOp, origOp.getLhs(), origOp.getRhs(), broadcastAttr,
                                            /*postOp=*/nullptr);

    return mlir::success();
}

}  // namespace

namespace vpux::arch30xx {

void ConvertLayers2VPUStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();
    patterns.add<EmbeddingSegmentsSumRewriter>(ctx, log);
    patterns.add<EmbeddingBagOffsetsSumRewriter>(ctx, log);
    patterns.add<AccumulateRewrite>(ctx, log);
}

}  // namespace vpux::arch30xx
