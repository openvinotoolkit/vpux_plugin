//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/attributes/strides.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux {
namespace VPUIP {

MemDimArr getStridesMemDims(vpux::NDTypeInterface tensorType);

}  // namespace VPUIP
}  // namespace vpux
