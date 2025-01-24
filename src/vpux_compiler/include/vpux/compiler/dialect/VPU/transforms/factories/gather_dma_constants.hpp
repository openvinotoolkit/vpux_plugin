//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

namespace vpux {
namespace VPU {

size_t getGatherDMAMaxIndicesListLength(VPU::ArchKind arch);
size_t getGatherDMAMaxElementSize(VPU::ArchKind arch);

}  // namespace VPU
}  // namespace vpux
