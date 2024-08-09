//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/types.hpp"
#include "vpux/compiler/dialect/VPUASM/dialect.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include <llvm/Support/Debug.h>

using namespace vpux;

//
// Dialect Hooks for BufferSection as a TypeParameter
//

mlir::AsmPrinter& operator<<(mlir::AsmPrinter& p, vpux::VPURT::BufferSection sec) {
    p << VPURT::stringifyBufferSection(sec);
    return p;
}

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/dialect/VPUASM/types.cpp.inc>
#undef GET_TYPEDEF_CLASSES

//
// register Types
//

void vpux::VPUASM::VPUASMDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/dialect/VPUASM/types.cpp.inc>
            >();
}
