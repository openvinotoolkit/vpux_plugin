//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

namespace vpux {
namespace arch37xx {

//
// AveragePoolRewriter
//
class AveragePoolRewriter final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    AveragePoolRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _log(log) {
        setDebugName("AveragePoolRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

}  // namespace arch37xx
}  // namespace vpux
