//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <cstdint>

namespace vpux {

// SwKernelOp::inputDimsMap will contain ABSENT_DIMS_FLAG to indicate a static shape
// In case of dynamic shape, inputDimsMap[inputIndex] will contain an index into SwKernelOp::inputDims range.
constexpr int32_t ABSENT_DIMS_FLAG = -1;

}  // namespace vpux
