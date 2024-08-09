//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/logger.hpp"

#include <memory>

namespace vpux {
namespace VPU {

//
// NCEConvolutionRewriter
//

class NCEConvolutionRewriter final : public mlir::OpRewritePattern<NCEConvolutionOp> {
public:
    NCEConvolutionRewriter(mlir::MLIRContext* ctx,
                           llvm::DenseMap<mlir::OpResult, OverlapDistributionParams>& overlapParamsLookup,
                           bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCEConvolutionOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log),
              _overlapParamsLookup(overlapParamsLookup) {
        setDebugName("NCEConvolutionRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
    llvm::DenseMap<mlir::OpResult, OverlapDistributionParams>& _overlapParamsLookup;
};

//
// NCEDepthConvolutionRewriter
//

class NCEDepthConvolutionRewriter final : public mlir::OpRewritePattern<NCEDepthConvolutionOp> {
public:
    NCEDepthConvolutionRewriter(mlir::MLIRContext* ctx,
                                llvm::DenseMap<mlir::OpResult, OverlapDistributionParams>& overlapParamsLookup,
                                bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCEDepthConvolutionOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log),
              _overlapParamsLookup(overlapParamsLookup) {
        setDebugName("NCEDepthConvolutionRewriter");
    }

public:
    bool _enableExplicitDistributedTensorAttr = false;
    mlir::LogicalResult matchAndRewrite(NCEDepthConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    llvm::DenseMap<mlir::OpResult, OverlapDistributionParams>& _overlapParamsLookup;
};

//
// NCEMaxPoolRewriter
//

class NCEMaxPoolRewriter final : public mlir::OpRewritePattern<NCEMaxPoolOp> {
public:
    NCEMaxPoolRewriter(mlir::MLIRContext* ctx,
                       llvm::DenseMap<mlir::OpResult, OverlapDistributionParams>& overlapParamsLookup,
                       bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCEMaxPoolOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log),
              _overlapParamsLookup(overlapParamsLookup) {
        setDebugName("NCEMaxPoolRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEMaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
    llvm::DenseMap<mlir::OpResult, OverlapDistributionParams>& _overlapParamsLookup;
};

//
// NCEAveragePoolRewriter
//

class NCEAveragePoolRewriter final : public mlir::OpRewritePattern<NCEAveragePoolOp> {
public:
    NCEAveragePoolRewriter(mlir::MLIRContext* ctx,
                           llvm::DenseMap<mlir::OpResult, OverlapDistributionParams>& overlapParamsLookup,
                           bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCEAveragePoolOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log),
              _overlapParamsLookup(overlapParamsLookup) {
        setDebugName("NCEAveragePoolRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEAveragePoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
    llvm::DenseMap<mlir::OpResult, OverlapDistributionParams>& _overlapParamsLookup;
};

//
// NCEEltwiseRewriterRewrite
//

class NCEEltwiseRewriter final : public mlir::OpRewritePattern<NCEEltwiseOp> {
public:
    NCEEltwiseRewriter(mlir::MLIRContext* ctx,
                       llvm::DenseMap<mlir::OpResult, OverlapDistributionParams>& overlapParamsLookup,
                       bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCEEltwiseOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log),
              _overlapParamsLookup(overlapParamsLookup) {
        setDebugName("NCEEltwiseRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEEltwiseOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
    llvm::DenseMap<mlir::OpResult, OverlapDistributionParams>& _overlapParamsLookup;
};

//
// NCESWRewriter
//

class NCESWRewriter final : public mlir::OpInterfaceRewritePattern<VPU::SWOpInterface> {
public:
    NCESWRewriter(mlir::MLIRContext* ctx,
                  llvm::DenseMap<mlir::OpResult, OverlapDistributionParams>& overlapParamsLookup,
                  bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::SWOpInterface>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log),
              _overlapParamsLookup(overlapParamsLookup) {
        setDebugName("NCESWRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::SWOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
    llvm::DenseMap<mlir::OpResult, OverlapDistributionParams>& _overlapParamsLookup;
};

//
// NCECompressConvolutionRewriterRewrite
//

class NCECompressConvolutionRewriter final : public mlir::OpRewritePattern<NCECompressConvolutionOp> {
public:
    NCECompressConvolutionRewriter(mlir::MLIRContext* ctx,
                                   llvm::DenseMap<mlir::OpResult, OverlapDistributionParams>& overlapParamsLookup,
                                   bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCECompressConvolutionOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log),
              _overlapParamsLookup(overlapParamsLookup) {
        setDebugName("NCECompressConvolutionRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCECompressConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
    llvm::DenseMap<mlir::OpResult, OverlapDistributionParams>& _overlapParamsLookup;
};

//
// NCEInterpolateRewriter
//

class NCEInterpolateRewriter final : public mlir::OpRewritePattern<NCEInterpolateOp> {
public:
    NCEInterpolateRewriter(mlir::MLIRContext* ctx,
                           llvm::DenseMap<mlir::OpResult, OverlapDistributionParams>& overlapParamsLookup,
                           bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCEInterpolateOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log),
              _overlapParamsLookup(overlapParamsLookup) {
        setDebugName("NCEInterpolateRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEInterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
    llvm::DenseMap<mlir::OpResult, OverlapDistributionParams>& _overlapParamsLookup;
};

//
// NCEMatMulRewriter
//

class NCEMatMulRewriter final : public mlir::OpRewritePattern<NCEMatMulOp> {
public:
    NCEMatMulRewriter(mlir::MLIRContext* ctx,
                      llvm::DenseMap<mlir::OpResult, OverlapDistributionParams>& overlapParamsLookup,
                      bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCEMatMulOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log),
              _overlapParamsLookup(overlapParamsLookup) {
        setDebugName("NCEMatMulRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEMatMulOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
    llvm::DenseMap<mlir::OpResult, OverlapDistributionParams>& _overlapParamsLookup;
};

}  // namespace VPU
}  // namespace vpux
