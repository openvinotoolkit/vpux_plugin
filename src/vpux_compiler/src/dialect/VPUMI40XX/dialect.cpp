//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/ops.hpp"

using namespace vpux;

//
// initialize
//

void vpux::VPUMI40XX::VPUMI40XXDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/dialect/VPUMI40XX/ops.cpp.inc>
            >();

    registerAttributes();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUMI40XX/dialect.cpp.inc>
