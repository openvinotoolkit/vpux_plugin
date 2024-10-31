//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/const/utils/content.hpp"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinTypes.h>

namespace vpux {
namespace Const {
namespace details {

//
// PositionRequirement
//
// Determines whether a constant transformation has a requirement on where to be placed in the
// list of transformations (e.g. if it should be the last transformation)
//

enum class PositionRequirement : int {
    NONE = 0,            // can be anywhere in the list
    PREFERRED_LAST = 1,  // will be last unless a transformation with LAST requirement is present
    LAST = 2             // will be the last transformation in the list
};

}  // namespace details
}  // namespace Const
}  // namespace vpux

//
// Generated
//

#include <vpux/compiler/dialect/const/attr_interfaces.hpp.inc>
