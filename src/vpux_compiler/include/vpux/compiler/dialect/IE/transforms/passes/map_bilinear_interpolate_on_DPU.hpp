//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"

namespace vpux {
namespace IE {

bool isLegalInterpolateOp(IE::InterpolateOp op, bool interpolateAsSEOp, LogCb logCb);

//
// MapBilinearInterpolateOnDPUBaseRewriter
//

class MapBilinearInterpolateOnDPUBaseRewriter : public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    MapBilinearInterpolateOnDPUBaseRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::InterpolateOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

protected:
    virtual mlir::Value createIdentityPooling(mlir::PatternRewriter& rewriter, mlir::Location loc,
                                              mlir::Value input) const;

private:
    mlir::Value scaleOnAxis(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input, int64_t inputSize,
                            int64_t outputSize, vpux::Dim axis, IE::MapCoordFuncT mapCoord) const;

private:
    Logger _log;
};

}  // namespace IE
}  // namespace vpux
