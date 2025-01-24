//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/types.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>

namespace vpux {
namespace VPUASM {

vpux::VPURT::BufferSection getBufferLocation(mlir::Operation* symTableOp, mlir::SymbolRefAttr symRef,
                                             Logger log = Logger::global());
vpux::VPURT::BufferSection getBufferLocation(ELF::SymbolReferenceMap& symRefMap, mlir::SymbolRefAttr symRef,
                                             Logger log = Logger::global());
vpux::VPUASM::BufferType getBufferType(ELF::SymbolReferenceMap& symRefMap, mlir::SymbolRefAttr symRef);

}  // namespace VPUASM
}  // namespace vpux
