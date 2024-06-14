//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"

using namespace vpux;
using namespace vpux::VPUMI40XX;
using namespace mlir;

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUMI40XX/ops.cpp.inc>

mlir::LogicalResult vpux::VPUMI40XX::BootstrapOp::verify() {
    return ::mlir::success();
}
