//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"

#include <mlir/IR/Dialect.h>

namespace vpux::VPUIPDPU::arch37xx {

mlir::LogicalResult verifyMPECfgOp(VPUIPDPU::MPECfgOp op);
mlir::LogicalResult verifyODUCfgOp(VPUIPDPU::ODUCfgOp op);
mlir::LogicalResult verifyDPUVariantOp(VPUIPDPU::DPUVariantOp op);

void registerVerifiersOpInterfaces(mlir::DialectRegistry& registry);

}  // namespace vpux::VPUIPDPU::arch37xx
