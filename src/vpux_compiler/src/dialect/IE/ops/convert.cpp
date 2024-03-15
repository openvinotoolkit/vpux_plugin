//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::ConvertOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ConvertOpAdaptor cvt(operands, attrs);
    if (mlir::failed(cvt.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = cvt.getInput().getType().cast<mlir::RankedTensorType>();
    const auto dstElemType = cvt.getDstElemType();

    inferredReturnShapes.emplace_back(inType.getShape(), dstElemType);
    return mlir::success();
}

bool vpux::IE::ConvertOp::areCastCompatible(mlir::TypeRange inputs, mlir::TypeRange outputs) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        return false;
    }

    const auto input = inputs.front().dyn_cast<vpux::NDTypeInterface>();
    const auto output = outputs.front().dyn_cast<vpux::NDTypeInterface>();

    if (!input || !output || input.getShape() != output.getShape()) {
        return false;
    }

    return true;
}

namespace {

#include <vpux/compiler/dialect/IE/convert.hpp.inc>

}  // namespace

void vpux::IE::ConvertOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext*) {
    populateWithGenerated(patterns);
}

mlir::OpFoldResult vpux::IE::ConvertOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    VPUX_THROW_UNLESS(operands.size() == 1, "Wrong number of operands : {0}", operands.size());

    if (const auto attr = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        return attr.convertElemType(getDstElemType());
    }

    return nullptr;
}
