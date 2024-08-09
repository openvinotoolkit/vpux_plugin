//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/attributes_utils.hpp"

namespace vpux {

int64_t getPositiveAxisInd(mlir::IntegerAttr axisIndAttr, int64_t rank) {
    auto axis = axisIndAttr.getValue().getSExtValue();

    if (axis < 0) {
        axis += rank;
    }

    return axis;
}

mlir::FailureOr<int64_t> getConstValue(mlir::Value input) {
    auto op = input.getDefiningOp<Const::DeclareOp>();
    if (op == nullptr) {
        return mlir::failure();
    }

    if (const auto attr = op.getContentAttr(); !attr.isSplat()) {
        return mlir::failure();
    }

    const auto content = op.getContent();
    return content.getSplatValue<int64_t>();
}

mlir::FailureOr<SmallVector<int64_t>> getConstArrValue(mlir::Value input) {
    auto op = input.getDefiningOp<Const::DeclareOp>();
    if (op == nullptr) {
        return mlir::failure();
    }

    const auto content = op.getContent();

    return to_small_vector(content.getValues<int64_t>());
}

mlir::FailureOr<int64_t> getConstOrAttrValue(mlir::Value input, mlir::IntegerAttr attr) {
    return (input != nullptr) ? getConstValue(input) : attr.getValue().getSExtValue();
}

mlir::FailureOr<SmallVector<int64_t>> getConstOrArrAttrValue(mlir::Value input, mlir::ArrayAttr attr) {
    return (input != nullptr) ? getConstArrValue(input) : parseIntArrayAttr<int64_t>(attr);
}

}  // namespace vpux
