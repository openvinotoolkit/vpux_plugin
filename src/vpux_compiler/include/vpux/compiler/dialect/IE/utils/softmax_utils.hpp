//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

namespace vpux {
namespace IE {

bool isSoftMaxAxisInLastMemDim(IE::SoftMaxOp op);

}  // namespace IE
}  // namespace vpux
