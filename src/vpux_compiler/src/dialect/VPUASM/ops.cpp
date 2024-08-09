//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"

using namespace vpux;

//
// initialize
//

void vpux::VPUASM::VPUASMDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPUASM/ops.cpp.inc>
            >();

    registerTypes();
    // registerAttributes();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUASM/dialect.cpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUASM/ops.cpp.inc>
