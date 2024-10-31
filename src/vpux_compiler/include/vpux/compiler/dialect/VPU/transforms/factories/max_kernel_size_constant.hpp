//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/max_kernel_size_constant.hpp"

namespace vpux {
namespace VPU {

VPU::MaxKernelSizeConstant getMaxKernelSizeConstant(VPU::ArchKind arch);

}  // namespace VPU
}  // namespace vpux
