//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/empty_node.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include "vpux/compiler/utils/infer_output_shape.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::MaxPool8Op::inferReturnTypes(mlir::MLIRContext* ctx,
                                                            std::optional<mlir::Location> optLoc,
                                                            mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                            mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                            mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::MaxPool8OpAdaptor maxPool8(operands, attrs, prop);
    if (mlir::failed(maxPool8.verify(loc))) {
        return mlir::failure();
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(maxPool8.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(maxPool8.getPadsBegin());
    const auto windowDilations = parseIntArrayAttr<int64_t>(maxPool8.getDilations());
    const auto windowShape = parseIntArrayAttr<int64_t>(maxPool8.getKernelSize());
    const auto windowStrides = parseIntArrayAttr<int64_t>(maxPool8.getStrides());
    const auto roundingType = maxPool8.getRoundingType();

    const auto inType = maxPool8.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape().raw();

    const auto shapeI64 = inferMaxPool8OutputShape(inShape, windowStrides, windowDilations, dataPaddingBelow,
                                                   dataPaddingAbove, windowShape, roundingType);

    const auto outType = inType.changeShape(Shape(shapeI64));
    inferredReturnTypes.push_back(outType);

    const auto outType1 = outType.changeElemType(maxPool8.getIndexElementType());
    inferredReturnTypes.push_back(outType1);

    return mlir::success();
}
