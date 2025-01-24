//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

#include "vpux/utils/core/numeric.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::MultiplyOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::MultiplyOpAdaptor multiply(operands, attrs, prop);
    if (mlir::failed(multiply.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = mlir::cast<mlir::RankedTensorType>(multiply.getInput1().getType());
    const auto in2Type = mlir::cast<mlir::RankedTensorType>(multiply.getInput2().getType());

    const auto outShapeRes =
            IE::broadcastEltwiseShape(in1Type.getShape(), in2Type.getShape(), multiply.getAutoBroadcast(), loc);

    if (mlir::succeeded(outShapeRes)) {
        auto outShapeResVec = outShapeRes.value();
        if (multiply.getOutputChannels().has_value()) {
            outShapeResVec[Dims4D::Act::C.ind()] = multiply.getOutputChannels().value();
        }

        const auto outOrder =
                in1Type.getRank() >= in2Type.getRank() ? vpux::getOrder(in1Type) : vpux::getOrder(in2Type);
        const auto outDesc = getTensorAttr(outOrder, getMemorySpace(in1Type), getBounds(in1Type));
        inferredReturnShapes.emplace_back(outShapeResVec, in1Type.getElementType(), outDesc);
    }

    return mlir::success();
}

mlir::OpFoldResult vpux::IE::MultiplyOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    VPUX_THROW_UNLESS(operands.size() == 2, "Wrong number of operands : {0}", operands.size());

    const bool shapeChanges = getShape(getInput1()) != getShape(getOutput());
    if (shapeChanges) {
        return nullptr;
    }

    const auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[1]);
    if (attr == nullptr || !attr.isSplat()) {
        return nullptr;
    }

    const auto content = static_cast<Const::ContentAttr>(attr).fold();
    return isDoubleEqual(content.getSplatValue<double>(), 1.0f) ? getInput1() : nullptr;
}
