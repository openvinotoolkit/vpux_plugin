//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::AccumulateOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::AccumulateOpAdaptor accumulate(operands, attrs);
    if (mlir::failed(accumulate.verify(loc))) {
        return mlir::failure();
    }

    const auto lhsType = accumulate.getLhs().getType().cast<mlir::ShapedType>();
    const auto rhsType = accumulate.getRhs().getType().cast<mlir::ShapedType>();
    VPUX_THROW_UNLESS(lhsType == rhsType, "Types of IE.Accumulate operands must match: lhs = {0}, rhs = {1}", lhsType,
                      rhsType);

    inferredReturnShapes.emplace_back(lhsType.getShape(), lhsType.getElementType());

    return mlir::success();
}

mlir::LogicalResult vpux::IE::AccumulateOp::verify() {
    const auto lhsType = getLhs().getType();
    const auto rhsType = getRhs().getType();
    if (lhsType != rhsType) {
        return errorAt(getLoc(), "Operation {0}: the type of operands must match. Got lhs = {1}, rhs = {2}",
                       getOperationName(), lhsType, rhsType);
    }

    const auto zeroScalesSet = getLhsScale() == nullptr && getRhsScale() == nullptr;
    const auto bothScalesSet = getLhsScale() != nullptr && getRhsScale() != nullptr;
    const auto supportedScales = zeroScalesSet || bothScalesSet;
    if (!supportedScales) {
        return errorAt(getLoc(), "Operation {0}: must set either both scales or none.", getOperationName());
    }

    return mlir::success();
}
