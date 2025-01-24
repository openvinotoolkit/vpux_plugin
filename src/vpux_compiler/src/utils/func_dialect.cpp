//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/func_dialect.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/range.hpp"

using namespace vpux;

mlir::func::FuncOp vpux::getCalledFunction(mlir::CallOpInterface callOp) {
    mlir::func::FuncOp funcOp = nullptr;
    auto symRefAttr = callOp.mlir::CallOpInterface::getCallableForCallee().dyn_cast<mlir::SymbolRefAttr>();
    if (symRefAttr != nullptr) {
        funcOp = mlir::dyn_cast_or_null<mlir::func::FuncOp>(
                mlir::SymbolTable::lookupNearestSymbolFrom(callOp, symRefAttr));
    }
    VPUX_THROW_WHEN(funcOp == nullptr, "Expected CallOp to a FuncOp");
    return funcOp;
}

SmallVector<mlir::func::CallOp> vpux::getCallSites(mlir::func::FuncOp funcOp, mlir::Operation* from) {
    auto useRangeOpt = funcOp.getSymbolUses(from);

    if (!useRangeOpt.has_value()) {
        return {};
    }

    auto callOps = to_small_vector(useRangeOpt.value() | transformed([](auto use) {
                                       return mlir::dyn_cast_or_null<mlir::func::CallOp>(use.getUser());
                                   }) |
                                   filtered([](mlir::func::CallOp callOp) {
                                       return callOp != nullptr;
                                   }));

    return callOps;
}
