//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/max_kernel_size_constant.hpp"

using namespace vpux::VPU;

int64_t MaxKernelSizeConstant::getMaxKernelSize() const {
    return self->getMaxKernelSize();
}
