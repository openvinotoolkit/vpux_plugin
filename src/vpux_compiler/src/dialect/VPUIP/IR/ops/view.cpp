
//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

using namespace vpux;

//
// ViewLikeOpInterface
//

mlir::Value vpux::VPUIP::ViewOp::getViewSource() {
    return getSource();
}
