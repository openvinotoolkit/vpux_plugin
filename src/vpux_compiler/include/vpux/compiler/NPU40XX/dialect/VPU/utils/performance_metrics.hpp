//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <cstdint>
#include "vpux/compiler/dialect/VPU/utils/performance_metrics.hpp"

namespace vpux::VPU::arch40xx {

vpux::VPU::FrequencyTable getFrequencyTable();

}  // namespace vpux::VPU::arch40xx
