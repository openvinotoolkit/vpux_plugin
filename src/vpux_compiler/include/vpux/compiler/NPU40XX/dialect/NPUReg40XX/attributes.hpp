//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

//
// Generated
//

#include <vpux/compiler/NPU40XX/dialect/NPUReg40XX/descriptors.hpp>
#include <vpux/compiler/NPU40XX/dialect/NPUReg40XX/enums.hpp.inc>

#define GET_ATTRDEF_CLASSES
#include <vpux/compiler/NPU40XX/dialect/NPUReg40XX/attributes.hpp.inc>
#undef GET_ATTRDEF_CLASSES
