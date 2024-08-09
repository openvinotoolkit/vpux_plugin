//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include <openvino/core/coordinate.hpp>
#include <openvino/op/group_conv.hpp>

using namespace vpux;

mlir::LogicalResult vpux::IE::GroupConvolutionBackpropDataOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::GroupConvolutionBackpropDataOpAdaptor groupConvBackpropData(operands, attrs, prop);
    if (mlir::failed(groupConvBackpropData.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = groupConvBackpropData.getInput().getType().cast<NDTypeInterface>();
    const auto inputShape = to_small_vector(inputType.getShape());
    const auto inputElemType = inputType.getElementType();
    const auto outputShape = groupConvBackpropData.getOutputShape();
    const auto filterShape =
            to_small_vector(groupConvBackpropData.getFilter().getType().cast<NDTypeInterface>().getShape());

    if (outputShape != nullptr) {
        return errorAt(loc, "Explicit output shape is not implemented");
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(groupConvBackpropData.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(groupConvBackpropData.getPadsBegin());
    const auto windowStrides = parseIntArrayAttr<int64_t>(groupConvBackpropData.getStrides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(groupConvBackpropData.getDilations());
    const auto outputPadding = parseIntArrayAttr<int64_t>(groupConvBackpropData.getOutputPadding());

    const auto mlirOutputShape = inferGroupConvBackpropOutputShape(
            inputShape, filterShape, windowStrides, dataPaddingBelow, dataPaddingAbove, windowDilations, outputPadding);
    inferredReturnShapes.emplace_back(mlirOutputShape, inputElemType);

    return mlir::success();
}
