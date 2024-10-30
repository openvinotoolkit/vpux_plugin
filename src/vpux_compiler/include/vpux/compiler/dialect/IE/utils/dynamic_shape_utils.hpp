//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/Operation.h>

namespace vpux::IE {

bool hasDynamicTensors(mlir::Operation* op);

}  // namespace vpux::IE
