//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

namespace vpux {
namespace IE {

mlir::LogicalResult genericBatchUnroll(mlir::Operation* origOp, size_t numInputs, mlir::PatternRewriter& rewriter);
bool isBatchEqualToOne(const mlir::Value val);
bool isShapeRankEqualToZero(const mlir::Value val);
bool areShapeRanksEqual(const mlir::Value lhs, const mlir::Value rhs);

//
// BatchUnrollConverter
//

template <class ConcreteOp>
class BatchUnrollConverter : public mlir::OpRewritePattern<ConcreteOp> {
public:
    BatchUnrollConverter(mlir::MLIRContext* ctx, Logger log, size_t numInputs)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log), _numInputs(numInputs) {
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final {
        return genericBatchUnroll(origOp.getOperation(), _numInputs, rewriter);
    }

private:
    Logger _log;

protected:
    size_t _numInputs;
};

}  // namespace IE
}  // namespace vpux
