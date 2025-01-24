//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ReverseOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                           mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                           mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ReverseOpAdaptor reverse(operands, attrs, prop);
    if (mlir::failed(reverse.verify(loc))) {
        return mlir::failure();
    }
    const auto dataType = mlir::cast<vpux::NDTypeInterface>(reverse.getInput().getType());
    const auto dataShape = dataType.getShape().raw();
    const auto elementType = dataType.getElementType();
    auto outType = dataType.changeElemType(elementType);
    outType = outType.changeShape(Shape(dataShape));

    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
