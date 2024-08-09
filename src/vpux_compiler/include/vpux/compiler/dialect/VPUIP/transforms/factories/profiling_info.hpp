//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/Types.h>
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

namespace vpux::VPUIP {

using TimestampTypeCb = mlir::Type (*)(mlir::MLIRContext* ctx);
using SetWorkloadIdsCb = void (*)(VPUIP::NCEClusterTaskOp nceClusterTaskOp);

TimestampTypeCb getTimestampTypeCb(VPU::ArchKind arch);
SetWorkloadIdsCb setWorkloadsIdsCb(VPU::ArchKind arch);

}  // namespace vpux::VPUIP
