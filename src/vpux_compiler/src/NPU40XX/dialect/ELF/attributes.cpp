//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/ELF/attributes.hpp"
#include "vpux/compiler/NPU40XX/dialect/ELF/dialect.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;

//
// Generated
//

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/NPU40XX/dialect/ELF/attributes.cpp.inc>
#include <vpux/compiler/NPU40XX/dialect/ELF/enums.cpp.inc>

//
// Dialect hooks
//

void ELF::ELFDialect::registerAttributes() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include <vpux/compiler/NPU40XX/dialect/ELF/attributes.cpp.inc>
            >();
}
