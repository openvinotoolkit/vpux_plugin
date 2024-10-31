//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/max_kernel_size_constant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

#include "vpux/utils/core/numeric.hpp"

using namespace vpux::VPU::arch37xx;

int64_t MaxKernelSizeConstant::getMaxKernelSize() const {
    return maxKernelSize;
}
