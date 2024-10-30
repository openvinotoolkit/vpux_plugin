//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/empty_node.hpp"

#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::MaxPool8Op::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::MaxPool8OpAdaptor maxPool8(operands, attrs, prop);
    if (mlir::failed(maxPool8.verify(loc))) {
        return mlir::failure();
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(maxPool8.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(maxPool8.getPadsBegin());
    const auto windowShape = parseIntArrayAttr<int64_t>(maxPool8.getKernelSize());
    const auto windowStrides = parseIntArrayAttr<int64_t>(maxPool8.getStrides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(maxPool8.getDilations());
    const auto roundingType = maxPool8.getRoundingType();

    const auto inType = maxPool8.getInput().getType().cast<mlir::ShapedType>().getElementType();
    const auto inShape = maxPool8.getInput().getType().cast<mlir::ShapedType>().getShape();

    auto outputShape = inferMaxPool8OutputShape(inShape, windowStrides, windowDilations, dataPaddingBelow,
                                                dataPaddingAbove, windowShape, roundingType);

    inferredReturnShapes.emplace_back(outputShape, inType);
    inferredReturnShapes.emplace_back(outputShape, maxPool8.getIndexElementType());

    return mlir::success();
}
