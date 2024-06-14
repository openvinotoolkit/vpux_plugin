//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux/compiler/dialect/VPUMI40XX/ops.hpp>
#include <vpux/compiler/dialect/VPUMI40XX/utils.hpp>

using namespace vpux;
using namespace VPUMI40XX;

//
// ActKernelInvocationOp
//

mlir::LogicalResult ActKernelInvocationOp::verify() {
    if (!isConfigureBarrierOpType(getWaitBarriers())) {
        return errorAt(getLoc(), "Operation {0}: waitBarriers should be of type ConfigureBarrier ", getOperationName());
    }

    if (!isConfigureBarrierOpType(getUpdateBarriers())) {
        return errorAt(getLoc(), "Operation {0}: updateBarriers should be of type ConfigureBarrier ",
                       getOperationName());
    }

    return ::mlir::success();
}

DotNodeColor VPUMI40XX::ActKernelInvocationOp::getNodeColor() {
    return DotNodeColor::ORANGE;
}

bool VPUMI40XX::ActKernelInvocationOp::printAttributes(llvm::raw_ostream& os, llvm::StringRef head,
                                                       llvm::StringRef middle, llvm::StringRef end) {
    printIndex(os, getType(), head, middle, end);
    return true;
}

DOT::EdgeDir VPUMI40XX::ActKernelInvocationOp::getEdgeDirection(mlir::Operation* source) {
    if (checkBarrierProductionRelationship(source, mlir::cast<VPUMI40XX::ExecutableTaskOpInterface>(getOperation()))) {
        return DOT::EdgeDir::EDGE_REVERSE;
    }
    return DOT::EdgeDir::EDGE_NORMAL;
}

bool ActKernelInvocationOp::supportsHardLink() {
    return false;
}
