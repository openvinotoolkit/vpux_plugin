//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/IR/BuiltinTypes.h>
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/utils/range_bound.hpp"

using namespace vpux;

mlir::LogicalResult VPU::RangeOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                   mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                   mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                   mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::RangeOpAdaptor range(operands, attrs, prop);
    if (mlir::failed(range.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = range.getStart().getType().cast<NDTypeInterface>();

    const auto outShape = Shape{mlir::ShapedType::kDynamic};
    const auto bounds = SmallVector<int64_t>{RANGEBOUND};

    const auto typeComponents = TypeComponents()
                                        .setShape(outShape)
                                        .setDimsOrder(DimsOrder::fromNumDims(outShape.size()))
                                        .setElementType(range.getDstElemType())
                                        .setBounds(getIntArrayAttr(ctx, bounds));

    auto outType = inType.changeTypeComponents(typeComponents);

    inferredReturnTypes.emplace_back(outType);

    return mlir::success();
}

//
// verify
//

mlir::LogicalResult vpux::VPU::RangeOp::verify() {
    const auto shape = getShape(getStart());

    if (shape.size() > 1) {
        return errorAt(*this, "Range kernel supports only up to 1D shapes, got '{0}'", shape.size());
    }

    return mlir::success();
}
