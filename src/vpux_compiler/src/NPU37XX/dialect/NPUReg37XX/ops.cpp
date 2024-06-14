//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/NPUReg37XX/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

using namespace vpux;

//
// initialize
//

void vpux::NPUReg37XX::NPUReg37XXDialect::initialize() {
    registerTypes();
}

//
// Generated
//

#include <vpux/compiler/NPU37XX/dialect/NPUReg37XX/dialect.cpp.inc>
