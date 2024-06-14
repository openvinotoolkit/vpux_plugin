//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/NPUReg37XX/types.hpp"
#include "vpux/compiler/NPU37XX/dialect/NPUReg37XX/ops.hpp"
#include "vpux/compiler/dialect/VPURegMapped/types.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/NPU37XX/dialect/NPUReg37XX/types.cpp.inc>
#undef GET_TYPEDEF_CLASSES

//
// register Types
//

void vpux::NPUReg37XX::NPUReg37XXDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include <vpux/compiler/NPU37XX/dialect/NPUReg37XX/types.cpp.inc>
#undef GET_TYPEDEF_LIST
            >();
}
