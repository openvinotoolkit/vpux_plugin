//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/types.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/attributes.hpp"
#include "vpux/compiler/utils/ELF/utils.hpp"

#include <npu_40xx_nnrt.hpp>

namespace vpux {
namespace NPUReg40XX {

const auto CMX_TILE_SELECT_OFFSET = 21;

uint32_t getTileSelectMaskForBuffer(VPUASM::DeclareBufferOp buffer);
uint32_t getTileSelectMaskForBuffer(VPUASM::DeclareTaskBufferOp taskBuffer);

template <class OpType>
OpType getOpFrom(ELF::SymbolReferenceMap& _symRefMap, std::optional<mlir::SymbolRefAttr> attr);

uint32_t getKernelEntry(ELF::SymbolReferenceMap& _symRefMap, std::optional<mlir::SymbolRefAttr> attr);
uint64_t getKernelTextSize(ELF::SymbolReferenceMap& _symRefMap, std::optional<mlir::SymbolRefAttr> attr);
llvm::StringRef getKernelPath(ELF::SymbolReferenceMap& _symRefMap, std::optional<mlir::SymbolRefAttr> kernelPath,
                              mlir::SymbolRefAttr taskType);
npu40xx::nn_public::VpuActWLType getActWLType(mlir::SymbolRefAttr taskType);
}  // namespace NPUReg40XX
}  // namespace vpux
