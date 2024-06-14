//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/dilated_utils.hpp"

using namespace vpux;

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::ExpandDilatedOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ExpandDilatedOpAdaptor expandDilated(operands, attrs);
    if (mlir::failed(expandDilated.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = expandDilated.getInput().getType().dyn_cast<vpux::NDTypeInterface>();
    if (!inType) {
        return mlir::failure();
    }

    const auto dilations = parseIntArrayAttr<int64_t>(expandDilated.getDilations());
    const auto newType = getDilatedType(inType, ShapeRef(dilations)).cast<mlir::RankedTensorType>();
    inferredReturnShapes.emplace_back(newType.getShape(), newType.getElementType(), newType.getEncoding());

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ExpandDilatedOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    VPUX_THROW_UNLESS(!operands.empty(), "Wrong number of operands : {0}", operands.size());

    if (const auto attr = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        const auto dilationsVal = parseIntArrayAttr<int64_t>(getDilations());
        return attr.expandDilated(ShapeRef(dilationsVal));
    }

    return nullptr;
}
