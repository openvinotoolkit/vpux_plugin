//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/hw_settings.hpp"

using namespace vpux;
mlir::LogicalResult vpux::IE::MVNOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::MVNOpAdaptor mvn(operands, attrs, prop);
    if (mlir::failed(mvn.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mvn.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inShape = inType.getShape();
    if (inShape.size() != 4 && inShape.size() != 5) {
        return errorAt(loc, "First input tensor should have 4 or 5 dimensions");
    }

    const auto outDesc = vpux::getTensorAttr(ctx, inType.getDimsOrder(), inType.getMemSpace());
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType(), outDesc);

    return mlir::success();
}

//
// Returns 'true' if minimum required channels to run without MVN-decomposition fit CMX
// minimum channels :  1 if across_channels = false
//                   all if across_channels = true
//
bool vpux::IE::MVNOp::channelsFitIntoCMX() {
    const auto inType = getInput().getType().cast<NDTypeInterface>();
    const auto inShape = inType.getShape();
    const auto bpp = vpux::getElemTypeSize(inType.getElementType()).to<Byte>().count();

    if (getInternalReshape().has_value()) {
        // attr present only for big-MVN instances that don't fit CMX thus require decomposition
        return false;
    }

    auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();

    const auto heightWidthSize = inShape[Dims4D::Act::H] * inShape[Dims4D::Act::W] * bpp;
    auto layerSize = getAcrossChannels() ? heightWidthSize * inShape[Dims4D::Act::C] : heightWidthSize;
    layerSize += vpux::DEFAULT_CMX_ALIGNMENT;  // alignment tolerance per buffer
    layerSize *= 2;                            // also consider output size

    return layerSize < vpux::VPU::getTotalCMXSize(moduleOp).count();
}

//
// build
//

void vpux::IE::MVNOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input,
                            ::mlir::BoolAttr across_channels, ::mlir::BoolAttr normalize_variance,
                            ::mlir::FloatAttr eps) {
    build(builder, state, input.getType(), input, across_channels, normalize_variance, eps, {});
}
