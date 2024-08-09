//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ExpandOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                          mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                          mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                          mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ExpandOpAdaptor expand(operands, attrs, prop);
    if (mlir::failed(expand.verify(loc))) {
        return mlir::failure();
    }

    const auto padBegin = parseIntArrayAttr<int64_t>(expand.getPadsBegin());
    const auto padEnd = parseIntArrayAttr<int64_t>(expand.getPadsEnd());

    const auto inType = expand.getInput().getType().cast<vpux::NDTypeInterface>();

    const auto newType = inType.pad(ShapeRef(padBegin), ShapeRef(padEnd));
    inferredReturnTypes.push_back(newType);

    return mlir::success();
}
