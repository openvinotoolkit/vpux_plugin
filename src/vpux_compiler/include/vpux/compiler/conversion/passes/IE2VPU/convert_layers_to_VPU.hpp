//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/IRMapping.h>

namespace vpux {

//
// computeWeightForEmbeddingOp
//

void computeWeightForEmbeddingOp(mlir::MLIRContext* ctx, mlir::RankedTensorType& weightsTensorType,
                                 mlir::DenseElementsAttr& baseAttr, llvm::ArrayRef<int64_t> weightsShape,
                                 vpux::NDTypeInterface inType);

//
// IfRewrite
//

class IfRewrite final : public mlir::OpRewritePattern<IE::IfOp> {
public:
    IfRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::IfOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::IfOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// CTCGreedyDecoderSeqLenRewrite
//

class CTCGreedyDecoderSeqLenRewrite final : public mlir::OpRewritePattern<IE::CTCGreedyDecoderSeqLenOp> {
public:
    CTCGreedyDecoderSeqLenRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::CTCGreedyDecoderSeqLenOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::CTCGreedyDecoderSeqLenOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// ProposalRewrite
//

class ProposalRewrite final : public mlir::OpRewritePattern<IE::ProposalOp> {
public:
    ProposalRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ProposalOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ProposalOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// SplitRewrite
//

class SplitRewrite final : public mlir::OpRewritePattern<IE::SplitOp> {
public:
    SplitRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SplitOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SplitOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// StubRewrite
//

class StubRewrite final : public mlir::OpRewritePattern<IE::StubOp> {
public:
    StubRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::StubOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::StubOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// RewriteNonMaxSuppression
//

class NonMaxSuppressionRewrite final : public mlir::OpRewritePattern<IE::NonMaxSuppressionOp> {
public:
    NonMaxSuppressionRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::NonMaxSuppressionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::NonMaxSuppressionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// InterpolateRewrite
//

// IE.Interpolate -> VPU.Interpolate
class InterpolateRewrite : public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    InterpolateRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::InterpolateOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// TopKRewrite
//

class TopKRewrite final : public mlir::OpRewritePattern<IE::TopKOp> {
public:
    TopKRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::TopKOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TopKOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// NormalizeL2Rewrite
//

class NormalizeL2Rewrite final : public mlir::OpRewritePattern<IE::NormalizeL2Op> {
public:
    NormalizeL2Rewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::NormalizeL2Op>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::NormalizeL2Op origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// RewriteEmbeddingBagPackedSum
//

class EmbeddingBagPackedSumRewrite final : public mlir::OpRewritePattern<IE::EmbeddingBagPackedSumOp> {
public:
    EmbeddingBagPackedSumRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::EmbeddingBagPackedSumOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::EmbeddingBagPackedSumOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// RewriteGRUCell
//

class GRUCellRewrite final : public mlir::OpRewritePattern<IE::GRUCellOp> {
public:
    GRUCellRewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::GRUCellOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GRUCellOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// TransposedConvRewrite
//

class TransposedConvRewrite final : public mlir::OpRewritePattern<IE::TransposedConvolutionOp> {
public:
    TransposedConvRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::TransposedConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposedConvolutionOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

}  // namespace vpux
