//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

namespace vpux {
namespace VPU {

int64_t getMaxLstmHiddenSizeConstant(VPU::ArchKind arch, bool sequenceEnabled = false);

}  // namespace VPU
}  // namespace vpux
