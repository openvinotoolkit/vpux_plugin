//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include "openvino/op/group_conv.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::TransposedConvolutionOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::TransposedConvolutionOpAdaptor convBackpropData(operands, attrs);
    if (mlir::failed(convBackpropData.verify(loc))) {
        return mlir::failure();
    }

    const auto featureType = convBackpropData.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto featureShape = featureType.getShape().raw();
    const auto outputShape = convBackpropData.getOutputShape();
    const auto filterShape = convBackpropData.getFilter().getType().cast<vpux::NDTypeInterface>().getShape().raw();

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(convBackpropData.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(convBackpropData.getPadsBegin());
    const auto windowStrides = parseIntArrayAttr<int64_t>(convBackpropData.getStrides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(convBackpropData.getDilations());
    const auto outputPadding = parseIntArrayAttr<int64_t>(convBackpropData.getOutputPadding());

    if (outputShape != nullptr) {
        const SmallVector<ov::Dimension> nDataShape(std::next(featureShape.begin(), 2), featureShape.end());
        const SmallVector<ov::Dimension> nFilterShape(std::next(filterShape.begin(), 2), filterShape.end());

        auto outputShapeConst = outputShape.getDefiningOp<Const::DeclareOp>();
        if (outputShapeConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for output_shape");
        }

        const auto outputShapeContent = outputShapeConst.getContent();
        const auto outputShapeVals = outputShapeContent.getValues<int64_t>();

        SmallVector<int64_t> mlirOutputShape;
        mlirOutputShape.push_back(featureShape[Dims4D::Act::N.ind()]);
        mlirOutputShape.push_back(filterShape[Dims4D::Filter::OC.ind()]);
        std::copy(outputShapeVals.begin(), outputShapeVals.end(), std::back_inserter(mlirOutputShape));

        auto outType = featureType.changeShape(Shape(mlirOutputShape));
        inferredReturnTypes.push_back(outType);
    } else {
        const auto mlirOutputShape =
                inferTransposedConvBackpropOutputShape(featureShape, filterShape, windowStrides, dataPaddingBelow,
                                                       dataPaddingAbove, windowDilations, outputPadding);

        auto outType = featureType.changeShape(Shape(mlirOutputShape));
        inferredReturnTypes.push_back(outType);
    }

    return mlir::success();
}
