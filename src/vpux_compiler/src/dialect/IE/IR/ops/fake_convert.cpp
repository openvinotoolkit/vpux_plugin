//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/range.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::FakeConvertOp::verify() {
    const auto dstType = getDstType();
    if (!dstType.isFloat8E4M3FN() && !dstType.isFloat8E5M2()) {
        return errorAt(*this, "Unsupported FakeConvert destination type {0}", dstType);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::FakeConvertOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::FakeConvertOpAdaptor cvt(operands, attrs, prop);
    if (mlir::failed(cvt.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = cvt.getInput().getType().cast<mlir::RankedTensorType>();

    inferredReturnShapes.emplace_back(inputType.getShape(), inputType.getElementType());

    return mlir::success();
}
