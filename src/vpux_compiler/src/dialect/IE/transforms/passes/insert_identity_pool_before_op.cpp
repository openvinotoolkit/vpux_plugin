//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/insert_identity_pool_before_op.hpp"

#include <mlir/IR/IRMapping.h>

using namespace vpux;

bool vpux::IE::isEligiblePostOp(mlir::Operation* op, Logger log) {
    auto postOpInterface = op->getOperand(0).getDefiningOp<IE::LayerWithPostOpInterface>();
    if (postOpInterface != nullptr && !postOpInterface.getPostOp().has_value() &&
        postOpInterface->getResult(0).hasOneUse()) {
        log.trace("A PostOp at {0} has already got a suitable producer", op->getLoc());
        return false;
    }
    return true;
}

mlir::LogicalResult vpux::IE::genericIdInserter(mlir::Operation* concreteOp, const InsertIdFunctor& insertId,
                                                mlir::PatternRewriter& rewriter, Logger log) {
    mlir::Operation* identityOp = insertId(concreteOp, rewriter, log);
    if (identityOp == nullptr) {
        return mlir::failure();
    }

    mlir::IRMapping mapper;
    const SmallVector<mlir::Value> inputsToMap = {identityOp->getResult(0)};
    mapper.map(concreteOp->getOperands(), ArrayRef(inputsToMap));
    auto* newLayerOp = rewriter.clone(*concreteOp, mapper);
    rewriter.replaceOp(concreteOp, newLayerOp->getResult(0));

    return mlir::success();
}
