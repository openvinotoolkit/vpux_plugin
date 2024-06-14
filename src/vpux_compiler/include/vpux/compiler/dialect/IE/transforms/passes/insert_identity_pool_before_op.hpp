//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

namespace vpux {
namespace IE {

using InsertIdFunctor = std::function<mlir::Operation*(mlir::Operation*, mlir::PatternRewriter& rewriter, Logger log)>;
bool isEligiblePostOp(mlir::Operation* op, Logger log);
mlir::LogicalResult genericIdInserter(mlir::Operation* concreteOp, const InsertIdFunctor& insertId,
                                      mlir::PatternRewriter& rewriter, Logger log);

//
// InsertIdPoolRewriter
//

template <typename ConcreteOp>
class InsertIdPoolRewriter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    InsertIdPoolRewriter(mlir::MLIRContext* ctx, const InsertIdFunctor& inserter, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _inserter(inserter), _log(log) {
        this->setDebugName("InsertIdPoolRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(ConcreteOp concreteOp, mlir::PatternRewriter& rewriter) const final {
        return genericIdInserter(concreteOp.getOperation(), _inserter, rewriter, _log);
    }

private:
    const InsertIdFunctor _inserter;
    Logger _log;
};

}  // namespace IE
}  // namespace vpux
