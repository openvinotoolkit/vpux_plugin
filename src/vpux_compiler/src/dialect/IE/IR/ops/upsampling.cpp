//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::UpsamplingOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::UpsamplingOpAdaptor upsampling(operands, attrs);
    if (mlir::failed(upsampling.verify(loc))) {
        return mlir::failure();
    }

    auto padChannelVector = parseIntArrayAttr<int32_t>(upsampling.getPadAttr().getPadsChannel());
    auto padHeightVector = parseIntArrayAttr<int32_t>(upsampling.getPadAttr().getPadsHeight());
    auto padWidthVector = parseIntArrayAttr<int32_t>(upsampling.getPadAttr().getPadsWidth());
    auto upsamplingFactorVector = parseIntArrayAttr<int32_t>(upsampling.getUpsamplingFactor());

    const auto inType = upsampling.getInput().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();

    VPUX_THROW_UNLESS(inShape.size() == 4, "Upsampling supports only 4D input tensor");
    VPUX_THROW_UNLESS(padChannelVector.size() == 2, "Upsampling supports pad channel on both sides");
    VPUX_THROW_UNLESS(padHeightVector.size() == 2, "Upsampling supports pad height on both sides");
    VPUX_THROW_UNLESS(padWidthVector.size() == 2, "Upsampling supports pad width on both sides");
    VPUX_THROW_UNLESS(upsamplingFactorVector.size() == 3, "Upsampling supports factors only for 3 axes");

    SmallVector<int64_t> outputShape{
            inShape[0],
            inShape[1] + (inShape[1] - 1) * (upsamplingFactorVector[2] - 1) + padChannelVector[0] + padChannelVector[1],
            inShape[2] + (inShape[2] - 1) * (upsamplingFactorVector[1] - 1) + padHeightVector[0] + padHeightVector[1],
            inShape[3] + (inShape[3] - 1) * (upsamplingFactorVector[0] - 1) + padWidthVector[0] + padWidthVector[1],
    };

    inferredReturnShapes.emplace_back(outputShape, inType.getElementType());

    return mlir::success();
}
