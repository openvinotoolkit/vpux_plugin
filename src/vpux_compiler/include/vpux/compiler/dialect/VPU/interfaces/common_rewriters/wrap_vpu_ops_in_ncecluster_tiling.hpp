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
    NCEConvolutionRewriter(mlir::MLIRContext* ctx, bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCEConvolutionOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log) {
        setDebugName("NCEConvolutionRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
};

//
// NCEDepthConvolutionRewriter
//

class NCEDepthConvolutionRewriter final : public mlir::OpRewritePattern<NCEDepthConvolutionOp> {
public:
    NCEDepthConvolutionRewriter(mlir::MLIRContext* ctx, bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCEDepthConvolutionOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log) {
        setDebugName("NCEDepthConvolutionRewriter");
    }

public:
    bool _enableExplicitDistributedTensorAttr = false;
    mlir::LogicalResult matchAndRewrite(NCEDepthConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// NCEMaxPoolRewriter
//

class NCEMaxPoolRewriter final : public mlir::OpRewritePattern<NCEMaxPoolOp> {
public:
    NCEMaxPoolRewriter(mlir::MLIRContext* ctx, bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCEMaxPoolOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log) {
        setDebugName("NCEMaxPoolRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEMaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
};

//
// NCEAveragePoolRewriter
//

class NCEAveragePoolRewriter final : public mlir::OpRewritePattern<NCEAveragePoolOp> {
public:
    NCEAveragePoolRewriter(mlir::MLIRContext* ctx, bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCEAveragePoolOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log) {
        setDebugName("NCEAveragePoolRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEAveragePoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
};

//
// NCEEltwiseRewriterRewrite
//

class NCEEltwiseRewriter final : public mlir::OpRewritePattern<NCEEltwiseOp> {
public:
    NCEEltwiseRewriter(mlir::MLIRContext* ctx, bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCEEltwiseOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log) {
        setDebugName("NCEEltwiseRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEEltwiseOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
};

//
// NCESWRewriter
//

class NCESWRewriter final : public mlir::OpInterfaceRewritePattern<VPU::SWOpInterface> {
public:
    NCESWRewriter(mlir::MLIRContext* ctx, bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::SWOpInterface>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log) {
        setDebugName("NCESWRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::SWOpInterface origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
};

//
// NCECompressConvolutionRewriterRewrite
//

class NCECompressConvolutionRewriter final : public mlir::OpRewritePattern<NCECompressConvolutionOp> {
public:
    NCECompressConvolutionRewriter(mlir::MLIRContext* ctx, bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCECompressConvolutionOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log) {
        setDebugName("NCECompressConvolutionRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCECompressConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
};

//
// NCEInterpolateRewriter
//

class NCEInterpolateRewriter final : public mlir::OpRewritePattern<NCEInterpolateOp> {
public:
    NCEInterpolateRewriter(mlir::MLIRContext* ctx, bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<NCEInterpolateOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log) {
        setDebugName("NCEInterpolateRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(NCEInterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
};

}  // namespace VPU
}  // namespace vpux
