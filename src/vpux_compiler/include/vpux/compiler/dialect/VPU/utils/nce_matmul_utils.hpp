//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

namespace vpux::VPU {

mlir::RankedTensorType inferNCEMatmulOutputType(vpux::NDTypeInterface input1Type, vpux::NDTypeInterface input2Type,
                                                vpux::NDTypeInterface origOutputType);

}  // namespace vpux::VPU
