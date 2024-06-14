//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPURegMapped/types.hpp"
#include "vpux/utils/core/string_ref.hpp"

#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "vpux/compiler/dialect/VPURegMapped/attributes.hpp"

//
// Generated
//

#define GET_TYPEDEF_CLASSES
#include <vpux/compiler/NPU40XX/dialect/NPUReg40XX/types.hpp.inc>
#undef GET_TYPEDEF_CLASSES
