//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI37XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/ops.hpp"

using namespace vpux;

//
// initialize
//

void vpux::VPUMI37XX::VPUMI37XXDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPUMI37XX/ops.cpp.inc>
            >();

    registerTypes();
    registerAttributes();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUMI37XX/dialect.cpp.inc>
