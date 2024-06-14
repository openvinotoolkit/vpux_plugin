//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/dialect.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"

//
// initialize
//

void vpux::ELF::ELFDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <vpux/compiler/NPU40XX/dialect/ELF/ops.cpp.inc>
            >();

    registerAttributes();
}

//
// Generated
//

#include <vpux/compiler/NPU40XX/dialect/ELF/dialect.cpp.inc>
