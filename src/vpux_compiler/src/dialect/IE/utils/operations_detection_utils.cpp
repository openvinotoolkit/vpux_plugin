//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/operations_detection_utils.hpp"

using namespace vpux;

void vpux::IE::detectAndReplacePreProcessedAgrs(Logger _log, mlir::func::FuncOp netFunc,
                                                SmallVector<mlir::Value>& newArgs, ArrayRef<mlir::Value> newResults) {
    if (newResults.size() != netFunc.getNumResults()) {
        // It is a check for potential multiple return statements.
        // They can appear in case of control flow:
        //
        // scf.if (%some_condition) {
        //    %0 = IE.Op1(...)
        //    return %0
        // } else {
        //     %1 = IE.Op2(...)
        //     return %1
        // }
        _log.trace("Mismatch between the number of post-processing operations and results, '{0}' != '{1}'",
                   newResults.size(), netFunc.getNumResults());
        return;
    }

    const auto getType = [](mlir::Value val) {
        return val.getType();
    };
    const auto newArgTypes = to_small_vector(newArgs | transformed(getType));
    const auto newResultsTypes = to_small_vector(newResults | transformed(getType));

    if (updateFunctionSignature(netFunc, newArgTypes, newResultsTypes, _log).failed()) {
        _log.trace("Failed to updateFunctionSignature");
        return;
    }

    _log.trace("Redirect arguments");

    for (const auto& p : netFunc.getArguments() | indexed) {
        const auto ind = p.index();

        auto curArg = p.value();
        auto newArg = newArgs[ind];

        if (curArg != newArg) {
            _log.nest().trace("Switch argument at index '{0}' from '{1}' to '{2}'", ind, curArg, newArg);

            curArg.setType(newArg.getType());
            newArg.replaceAllUsesWith(curArg);
        }
    }
}

void vpux::IE::detectAndReplacePostProcessedRes(Logger _log, mlir::func::FuncOp& netFunc,
                                                ArrayRef<mlir::Value> newResults) {
    netFunc.walk([&](mlir::func::ReturnOp retOp) {
        for (auto& curRes : retOp->getOpOperands()) {
            const auto& newRes = newResults[curRes.getOperandNumber()];

            if (curRes.get() != newRes) {
                _log.nest().trace("Switch result from '{1}' to '{2}'", curRes.get(), newRes);
                curRes.set(newRes);
            }
        }
    });
}
