//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux {
namespace IE {

void detectAndReplacePreProcessedAgrs(Logger _log, mlir::func::FuncOp netFunc, SmallVector<mlir::Value>& newArgs,
                                      ArrayRef<mlir::Value> newResults);

void detectAndReplacePostProcessedRes(Logger _log, mlir::func::FuncOp& netFunc, ArrayRef<mlir::Value> newResults);

template <class ConcreteOp1, class ConcreteOp2 = ConcreteOp1>
mlir::Value detectValueAfterOperation(mlir::Value val, SmallVector<mlir::Operation*>& ops) {
    if (!val.hasOneUse()) {
        return val;
    }

    if (auto concreteOp = mlir::dyn_cast<ConcreteOp1>(*val.user_begin())) {
        ops.push_back(concreteOp);
        return detectValueAfterOperation<ConcreteOp1, ConcreteOp2>(concreteOp.getOutput(), ops);
    }
    if (auto concreteOp = mlir::dyn_cast<ConcreteOp2>(*val.user_begin())) {
        ops.push_back(concreteOp);
        return detectValueAfterOperation<ConcreteOp1, ConcreteOp2>(concreteOp.getOutput(), ops);
    }

    return val;
}

template <class ConcreteOp1, class ConcreteOp2 = ConcreteOp1>
mlir::Value detectValueBeforeOperation(mlir::Value val, SmallVector<mlir::Operation*>& ops) {
    auto* producer = val.getDefiningOp();
    if (producer == nullptr) {
        return val;
    }

    if (auto concreteOp = mlir::dyn_cast<ConcreteOp1>(producer)) {
        ops.push_back(concreteOp);
        return detectValueBeforeOperation<ConcreteOp1, ConcreteOp2>(concreteOp.getInput(), ops);
    }
    if (auto concreteOp = mlir::dyn_cast<ConcreteOp2>(producer)) {
        ops.push_back(concreteOp);
        return detectValueBeforeOperation<ConcreteOp1, ConcreteOp2>(concreteOp.getInput(), ops);
    }

    return val;
}

}  // namespace IE
}  // namespace vpux
