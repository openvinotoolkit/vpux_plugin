//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::RandomUniformOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::RandomUniformOpAdaptor rand(operands, attrs, prop);
    if (mlir::failed(rand.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = rand.getMin().getType().cast<vpux::NDTypeInterface>();
    const auto shape = parseIntArrayAttr<int64_t>(rand.getOutputShape());
    auto outShapeType = inType.changeShape(Shape(shape));
    auto outType = outShapeType.changeElemType(rand.getOutputType());
    inferredReturnTypes.push_back(outType);
    return mlir::success();
}
