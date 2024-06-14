//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux {
namespace IE {

bool isLegalReorderAddPattern(IE::ReorderOp origOp);
bool isLegalReorderAvgPoolPattern(IE::ReorderOp origOp);

}  // namespace IE
}  // namespace vpux
