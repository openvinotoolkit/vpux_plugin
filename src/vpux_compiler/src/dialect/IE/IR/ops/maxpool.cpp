//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/empty_node.hpp"

#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::MaxPoolOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::MaxPoolOpAdaptor maxPool(operands, attrs, prop);
    if (mlir::failed(maxPool.verify(loc))) {
        return mlir::failure();
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(maxPool.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(maxPool.getPadsBegin());
    const auto windowShape = parseIntArrayAttr<int64_t>(maxPool.getKernelSize());
    const auto windowStrides = parseIntArrayAttr<int64_t>(maxPool.getStrides());
    const auto roundingType = maxPool.getRoundingType();

    const auto inType = maxPool.getInput().getType().cast<mlir::ShapedType>().getElementType();
    const auto inShape = maxPool.getInput().getType().cast<mlir::ShapedType>().getShape();

    auto outputShape = inferMaxPoolOutputShape(inShape, windowStrides, dataPaddingBelow, dataPaddingAbove, windowShape,
                                               roundingType);

    if (maxPool.getOutputChannels().has_value()) {
        outputShape[Dims4D::Act::C.ind()] = maxPool.getOutputChannels().value();
    }

    inferredReturnShapes.emplace_back(outputShape, inType);

    return mlir::success();
}
