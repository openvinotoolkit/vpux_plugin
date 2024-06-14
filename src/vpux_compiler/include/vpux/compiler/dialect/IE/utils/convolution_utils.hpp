//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

namespace vpux {
namespace IE {

mlir::LogicalResult canConvertGroupConvToConv(IE::GroupConvolutionOp groupconv, bool isAttrCheckEnabled = true);
bool groupConvIsEltwise(IE::GroupConvolutionOp convOp);

//
// FuseConvAndBias
//

class FuseConvAndBias final : public mlir::OpRewritePattern<IE::ScaleShiftOp> {
public:
    using mlir::OpRewritePattern<IE::ScaleShiftOp>::OpRewritePattern;

    void initialize() {
        setDebugName("FuseConvAndBias");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ScaleShiftOp biasOp, mlir::PatternRewriter& rewriter) const final;
};

}  // namespace IE
}  // namespace vpux
