//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

#include "vpux/utils/core/numeric.hpp"

using namespace vpux;

namespace {
mlir::AffineMap inferOrder(const mlir::RankedTensorType lhsType, const mlir::RankedTensorType rhsType) {
    return lhsType.getRank() > rhsType.getRank() ? vpux::getOrder(lhsType) : vpux::getOrder(rhsType);
}

SmallVector<int64_t> dispatchBounds(const mlir::Value operand) {
    auto boundsAttr = getBounds(operand);
    if (boundsAttr == nullptr) {
        return to_small_vector(getShape(operand));
    }
    return parseIntArrayAttr<int64_t>(boundsAttr);
}

mlir::ArrayAttr inferOutputBounds(const mlir::Value lhs, const mlir::Value rhs, const ShapeRef outputShape,
                                  const std::optional<int64_t> outputChannels,
                                  const IE::AutoBroadcastType autoBroadcast) {
    if (outputShape.isStatic()) {
        return nullptr;
    }
    const auto lhsBounds = dispatchBounds(lhs);
    const auto rhsBounds = dispatchBounds(rhs);
    auto outBoundsRes = IE::broadcastEltwiseShape(lhsBounds, rhsBounds, autoBroadcast, lhs.getLoc());
    if (mlir::failed(outBoundsRes)) {
        return nullptr;
    }

    auto outBoundsVec = outBoundsRes.value();
    const auto dimC = Dims4D::Act::C.ind();
    if (outputChannels.has_value() && dimC < checked_cast<int32_t>(outBoundsVec.size())) {
        outBoundsVec[dimC] = outputChannels.value();
    }
    return getIntArrayAttr(lhs.getContext(), outBoundsVec);
}
};  // namespace

mlir::LogicalResult vpux::IE::AddOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::AddOpAdaptor add(operands, attrs, prop);
    if (mlir::failed(add.verify(loc))) {
        return mlir::failure();
    }

    const auto in1Type = mlir::cast<mlir::RankedTensorType>(add.getInput1().getType());
    const auto in2Type = mlir::cast<mlir::RankedTensorType>(add.getInput2().getType());

    auto outShapeRes = IE::broadcastEltwiseShape(in1Type.getShape(), in2Type.getShape(), add.getAutoBroadcast(), loc);

    if (mlir::succeeded(outShapeRes)) {
        auto outShapeResVec = outShapeRes.value();
        const auto outBoundsAttr = inferOutputBounds(add.getInput1(), add.getInput2(), Shape(outShapeResVec),
                                                     add.getOutputChannels(), add.getAutoBroadcast());
        if (add.getOutputChannels().has_value()) {
            outShapeResVec[Dims4D::Act::C.ind()] = add.getOutputChannels().value();
        }

        const auto outDesc = vpux::getTensorAttr(inferOrder(in1Type, in2Type), /*memSpace=*/nullptr, outBoundsAttr);
        inferredReturnShapes.emplace_back(outShapeResVec, in1Type.getElementType(), outDesc);
    }

    return outShapeRes;
}

mlir::OpFoldResult vpux::IE::AddOp::fold(FoldAdaptor adaptor) {
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
    return isDoubleEqual(content.getSplatValue<double>(), 0.0f) ? getInput1() : nullptr;
}

mlir::LogicalResult vpux::IE::AddOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                       mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    auto loc = getLoc();

    auto outShape = IE::reifyEltwiseTensors(builder, getInput1(), getInput2(), getAutoBroadcast(), loc);

    if (mlir::failed(outShape)) {
        return outShape;
    }

    reifiedReturnShapes.emplace_back(std::move(outShape.value()));
    return mlir::success();
}
