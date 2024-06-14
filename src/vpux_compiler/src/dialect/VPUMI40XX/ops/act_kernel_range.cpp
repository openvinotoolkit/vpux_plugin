//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux/compiler/dialect/VPUMI40XX/ops.hpp>
#include <vpux/compiler/dialect/VPUMI40XX/utils.hpp>

using namespace vpux;
using namespace VPUMI40XX;

//
// ActKernelRangeOp
//

DotNodeColor VPUMI40XX::ActKernelRangeOp::getNodeColor() {
    return DotNodeColor::NONE;
}

bool VPUMI40XX::ActKernelRangeOp::printAttributes(llvm::raw_ostream& os, llvm::StringRef head, llvm::StringRef middle,
                                                  llvm::StringRef end) {
    printIndex(os, getType(), head, middle, end);
    return true;
}

DOT::EdgeDir VPUMI40XX::ActKernelRangeOp::getEdgeDirection(mlir::Operation*) {
    // don't really care about the ranges
    return DOT::EdgeDir::EDGE_SKIP;
}
