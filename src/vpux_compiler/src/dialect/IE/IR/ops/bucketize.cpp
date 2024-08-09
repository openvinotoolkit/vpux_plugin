//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::BucketizeOp::verify() {
    const mlir::Type constInt32 = getSInt32Type(getContext());
    const mlir::Type constInt64 = getSInt64Type(getContext());

    if (!(getOutputType() == constInt32 || getOutputType() == constInt64)) {
        return errorAt(*this,
                       "Attribute output_type support only SI32 and SI64 type according to schema definition for this "
                       "attribute. Got {0} type",
                       getOutputType());
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::BucketizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::BucketizeOpAdaptor bucketize(operands, attrs, prop);
    if (mlir::failed(bucketize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = bucketize.getData().getType().cast<mlir::ShapedType>();

    inferredReturnShapes.emplace_back(inType.getShape(), bucketize.getOutputType());

    return mlir::success();
}
