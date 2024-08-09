//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::DesparsifyOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              std::optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::OpaqueProperties prop,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DesparsifyOpAdaptor desparsify(operands, attrs, prop);
    if (mlir::failed(desparsify.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = desparsify.getInput().getType().cast<vpux::VPU::SparseTensorType>();
    const auto dataType = inType.getData();

    inferredReturnTypes.push_back(dataType);

    return mlir::success();
}
