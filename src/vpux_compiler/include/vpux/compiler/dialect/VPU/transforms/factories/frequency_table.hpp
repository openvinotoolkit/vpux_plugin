//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/performance_metrics.hpp"

namespace vpux {
namespace VPU {

using FrequencyTableCb = VPU::FrequencyTable (*)();

FrequencyTableCb getFrequencyTable(VPU::ArchKind arch);

}  // namespace VPU
}  // namespace vpux
