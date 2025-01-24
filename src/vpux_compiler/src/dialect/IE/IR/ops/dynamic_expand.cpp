//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/pad_extract.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::DynamicExpandOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::DynamicExpandOpAdaptor dynExpand(operands, attrs, prop);
    if (mlir::failed(dynExpand.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(dynExpand.getInput().getType());

    auto inShapeInfo = ShapeInfo::fromNDType(inType);
    inferredReturnShapes.emplace_back(inType.getShape().isDynamic() ? inShapeInfo.bounds : inShapeInfo.shape,
                                      inType.getElementType());
    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::DynamicExpandOp::fold(FoldAdaptor /*adaptor*/) {
    if (auto NDType = mlir::dyn_cast<NDTypeInterface>(getInput().getType())) {
        return NDType.getShape().isStatic() ? getInput() : nullptr;
    }
    return nullptr;
}
