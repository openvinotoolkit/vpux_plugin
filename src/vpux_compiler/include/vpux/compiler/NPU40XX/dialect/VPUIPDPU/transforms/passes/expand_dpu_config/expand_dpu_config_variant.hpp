//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

namespace vpux::VPUIPDPU::arch40xx {

mlir::LogicalResult buildDPUVariantIDU(VPUASM::DPUVariantOp origVarOp, mlir::OpBuilder& builder, const Logger& log,
                                       ELF::SymbolReferenceMap& symRefMap);

mlir::LogicalResult buildDPUVariantODU(VPUASM::DPUVariantOp origVarOp, mlir::OpBuilder& builder, const Logger& log,
                                       mlir::Block* varBlock, ELF::SymbolReferenceMap& symRefMap);

}  // namespace vpux::VPUIPDPU::arch40xx
