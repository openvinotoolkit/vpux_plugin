//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/rewriters/utils.hpp"

namespace vpux::VPUIPDPU::arch40xx {

mlir::LogicalResult buildDPUInvariantIDU(VPUASM::DPUInvariantOp origInvOp, mlir::OpBuilder& builder, const Logger& log,
                                         mlir::Block* invBlock,
                                         const std::unordered_map<BlockArg, size_t>& invBlockArgsPos);

mlir::LogicalResult buildDPUInvariantMPE(VPUASM::DPUInvariantOp origInvOp, mlir::OpBuilder& builder,
                                         mlir::Block* invBlock,
                                         const std::unordered_map<BlockArg, size_t>& invBlockArgsPos);

mlir::LogicalResult buildDPUInvariantPPE(VPUASM::DPUInvariantOp origInvOp, mlir::OpBuilder& builder, const Logger& log,
                                         mlir::Block* invBlock,
                                         const std::unordered_map<BlockArg, size_t>& invBlockArgsPos);

mlir::LogicalResult buildDPUInvariantODU(VPUASM::DPUInvariantOp origInvOp, mlir::OpBuilder& builder, const Logger& log,
                                         mlir::Block* invBlock,
                                         const std::unordered_map<BlockArg, size_t>& invBlockArgsPos,
                                         ELF::SymbolReferenceMap& symRefMap);

}  // namespace vpux::VPUIPDPU::arch40xx
