//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::IE::RandomUniformOp::verify() {
    int64_t numElements = 0;
    const auto hasOneElement = [&](mlir::Value tensor) {
        numElements = tensor.getType().cast<vpux::NDTypeInterface>().getNumElements();
        return numElements == 1;
    };

    if (!hasOneElement(getMin())) {
        return errorAt(*this, "Min should have only 1 element, while it has {0}", numElements);
    }

    if (!hasOneElement(getMax())) {
        return errorAt(*this, "Max should have only 1 element, while it has {0}", numElements);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::RandomUniformOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::RandomUniformOpAdaptor rand(operands, attrs);
    if (mlir::failed(rand.verify(loc))) {
        return mlir::failure();
    }

    const auto outShape = parseIntArrayAttr<int64_t>(rand.getOutputShape());
    inferredReturnShapes.emplace_back(outShape, rand.getOutputType());

    return mlir::success();
}
