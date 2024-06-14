//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/IE/locations.hpp"

#include <vpux/utils/core/error.hpp>
#include "vpux/utils/core/format.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>

mlir::Location vpux::IE::createLayerLocation(mlir::MLIRContext* ctx, const std::string& layerName,
                                             const std::string& layerType) {
    const auto layerNameAttr = mlir::StringAttr::get(ctx, layerName);
    const auto nameLoc = mlir::NameLoc::get(layerNameAttr);

    SmallVector<mlir::NamedAttribute> fields;
    fields.emplace_back(mlir::StringAttr::get(ctx, "type"), mlir::StringAttr::get(ctx, layerType));
    fields.emplace_back(mlir::StringAttr::get(ctx, "name"), layerNameAttr);
    auto metadata = mlir::DictionaryAttr::get(ctx, fields);

    return mlir::FusedLoc::get(ctx, {nameLoc}, metadata);
}

mlir::Location vpux::IE::getValueLocation(mlir::Value val) {
    // value is produced by real operation, so use it
    if (auto producerOp = val.getDefiningOp()) {
        return producerOp->getLoc();
    }
    // value is a block argument, so a function argument
    if (auto arg = val.dyn_cast<mlir::BlockArgument>()) {
        auto ownerOp = arg.getOwner()->getParentOp();
        auto maybeFuncOp = mlir::dyn_cast<mlir::func::FuncOp>(ownerOp);
        VPUX_THROW_WHEN(maybeFuncOp == nullptr,
                        "Invalid type of parent operation, expected to get mlir::func::FuncOp, but got {0}",
                        maybeFuncOp);
        auto moduleOp = getModuleOp(maybeFuncOp);
        IE::CNNNetworkOp netOp;
        IE::CNNNetworkOp::getFromModule(moduleOp, netOp, maybeFuncOp);
        auto inputsInfo = to_small_vector(netOp.getInputsInfo().getOps<IE::DataInfoOp>());

        const size_t inputNum = arg.getArgNumber();
        return inputsInfo[inputNum]->getLoc();
    }
    VPUX_THROW("Can't get location of '{0}'", val);
}
