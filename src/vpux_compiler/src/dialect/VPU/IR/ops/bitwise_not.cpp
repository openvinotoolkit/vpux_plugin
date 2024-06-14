//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::BitwiseNotOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              std::optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::OpaqueProperties, mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::BitwiseNotOpAdaptor bitwiseNot(operands, attrs);
    if (mlir::failed(bitwiseNot.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = bitwiseNot.getInput1().getType();
    inferredReturnTypes.push_back(inType);

    return mlir::success();
}
