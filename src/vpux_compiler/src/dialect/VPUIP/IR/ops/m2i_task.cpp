//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::M2ITaskOp::inferReturnTypes(mlir::MLIRContext* /*ctx*/,
                                                             std::optional<mlir::Location> /*optLoc*/,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties props,
                                                             mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredTypes) {
    VPUIP::M2ITaskOpAdaptor adaptor(operands, attrs, props);
    inferredTypes.push_back(adaptor.getOutputBuff().getType());
    if (adaptor.getProfilingData() != nullptr) {
        inferredTypes.push_back(adaptor.getProfilingData().getType());
    }
    return mlir::success();
}
