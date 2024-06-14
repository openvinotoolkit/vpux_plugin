//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux/compiler/dialect/VPUMI40XX/utils.hpp>
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops_interfaces.hpp"

using namespace vpux;
using namespace VPUMI40XX;

//
// Dot Printer
//

DotNodeColor VPUMI40XX::M2IOp::getNodeColor() {
    return DotNodeColor::RED;
}

bool VPUMI40XX::M2IOp::printAttributes(llvm::raw_ostream& os, llvm::StringRef head, llvm::StringRef middle,
                                       llvm::StringRef end) {
    printIndex(os, getType(), head, middle, end);
    return true;
}

DOT::EdgeDir VPUMI40XX::M2IOp::getEdgeDirection(mlir::Operation* source) {
    if (checkBarrierProductionRelationship(source, mlir::cast<VPUMI40XX::ExecutableTaskOpInterface>(getOperation()))) {
        return DOT::EdgeDir::EDGE_REVERSE;
    }
    return DOT::EdgeDir::EDGE_NORMAL;
}

bool M2IOp::supportsHardLink() {
    return true;
}
