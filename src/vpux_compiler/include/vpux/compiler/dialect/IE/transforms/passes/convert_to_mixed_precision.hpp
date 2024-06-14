//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/utils/VPU/ppe_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux {
namespace IE {

using SupportedMixedPrecisionFunctor = std::function<bool(mlir::Operation*, const bool isPReLUSupported, Logger log)>;

class FloatOutConvRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    FloatOutConvRewriter(mlir::MLIRContext* ctx, const SupportedMixedPrecisionFunctor& isMixPrecisionSupported,
                         Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx),
              _isMixPrecisionSupported(isMixPrecisionSupported),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp convolutionOp, mlir::PatternRewriter& rewriter) const final;

private:
    const SupportedMixedPrecisionFunctor _isMixPrecisionSupported;
    Logger _log;
};

class FloatOutGroupConvRewriter final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    FloatOutGroupConvRewriter(mlir::MLIRContext* ctx, const SupportedMixedPrecisionFunctor& isMixPrecisionSupported,
                              Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx),
              _isMixPrecisionSupported(isMixPrecisionSupported),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp groupConvolutionOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    const SupportedMixedPrecisionFunctor _isMixPrecisionSupported;
    Logger _log;
};

class FloatOutAddRewriter final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    FloatOutAddRewriter(mlir::MLIRContext* ctx, const SupportedMixedPrecisionFunctor& isMixPrecisionSupported,
                        const bool allowDifferentScales, Logger log)
            : mlir::OpRewritePattern<IE::AddOp>(ctx),
              _isMixPrecisionSupported(isMixPrecisionSupported),
              _allowDifferentScales(allowDifferentScales),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const SupportedMixedPrecisionFunctor _isMixPrecisionSupported;
    const bool _allowDifferentScales;
    Logger _log;
};

class FloatOutTransposedConvRewriter final : public mlir::OpRewritePattern<IE::TransposedConvolutionOp> {
public:
    FloatOutTransposedConvRewriter(mlir::MLIRContext* ctx,
                                   const SupportedMixedPrecisionFunctor& isMixPrecisionSupported, Logger log)
            : mlir::OpRewritePattern<IE::TransposedConvolutionOp>(ctx),
              _isMixPrecisionSupported(isMixPrecisionSupported),
              _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::TransposedConvolutionOp transposedConvOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    const SupportedMixedPrecisionFunctor _isMixPrecisionSupported;
    Logger _log;
};

}  // namespace IE
}  // namespace vpux
