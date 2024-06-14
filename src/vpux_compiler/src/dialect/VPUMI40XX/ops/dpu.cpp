//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux/compiler/dialect/VPUMI40XX/ops.hpp>
#include <vpux/compiler/dialect/VPUMI40XX/utils.hpp>

using namespace vpux;
using namespace VPUMI40XX;

//
// DPUInvariantOp
//

mlir::LogicalResult DPUInvariantOp::verify() {
    if (!isConfigureBarrierOpType(getWaitBarriers())) {
        return errorAt(getLoc(), "Operation {0}: waitBarriers should be of type ConfigureBarrier ", getOperationName());
    }

    if (!isConfigureBarrierOpType(getUpdateBarriers())) {
        return errorAt(getLoc(), "Operation {0}: updateBarriers should be of type ConfigureBarrier ",
                       getOperationName());
    }

    return ::mlir::success();
}

DotNodeColor DPUInvariantOp::getNodeColor() {
    return DotNodeColor::AQUA;
}

bool DPUInvariantOp::printAttributes(llvm::raw_ostream& os, llvm::StringRef head, llvm::StringRef middle,
                                     llvm::StringRef end) {
    printIndex(os, getType(), head, middle, end);
    return true;
}

DOT::EdgeDir DPUInvariantOp::getEdgeDirection(mlir::Operation* source) {
    if (checkBarrierProductionRelationship(source, mlir::cast<ExecutableTaskOpInterface>(getOperation()))) {
        return DOT::EdgeDir::EDGE_REVERSE;
    }
    return DOT::EdgeDir::EDGE_NORMAL;
}

bool DPUInvariantOp::supportsHardLink() {
    return false;
}

//
// DPUVariant
//

DotNodeColor DPUVariantOp::getNodeColor() {
    return DotNodeColor::AQUAMARINE;
}

bool DPUVariantOp::printAttributes(llvm::raw_ostream& os, llvm::StringRef head, llvm::StringRef middle,
                                   llvm::StringRef end) {
    printIndex(os, getType(), head, middle, end);
    return true;
}

DOT::EdgeDir DPUVariantOp::getEdgeDirection(mlir::Operation*) {
    return DOT::EdgeDir::EDGE_NORMAL;
}

bool DPUVariantOp::supportsHardLink() {
    return true;
}
