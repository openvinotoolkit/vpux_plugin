//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI40XX/attributes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/dialect.hpp"

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

using namespace vpux;

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/dialect/VPUMI40XX/attributes.cpp.inc>

#include <vpux/compiler/dialect/VPUMI40XX/enums.cpp.inc>

//
// Dialect hooks
//

void vpux::VPUMI40XX::VPUMI40XXDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/dialect/VPUMI40XX/attributes.cpp.inc>
            >();
}
