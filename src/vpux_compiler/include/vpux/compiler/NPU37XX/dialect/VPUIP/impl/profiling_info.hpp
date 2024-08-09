//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/Types.h>
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

using namespace vpux;

namespace vpux::VPUIP::arch37xx {

mlir::Type getTimestampType(mlir::MLIRContext* ctx);
void setWorkloadIds(VPUIP::NCEClusterTaskOp nceClusterTaskOp);

}  // namespace vpux::VPUIP::arch37xx
