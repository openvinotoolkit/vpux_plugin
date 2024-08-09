//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/utils/performance_metrics.hpp"

namespace {

// Base of frequency values used in tables (in MHz).
static constexpr uint32_t FREQ_BASE = 700;
// Step of frequency for each entry in tables (in MHz).
static constexpr uint32_t FREQ_STEP = 150;

}  // namespace

vpux::VPU::FrequencyTable vpux::VPU::arch37xx::getFrequencyTable() {
    return vpux::VPU::FrequencyTable{FREQ_BASE, FREQ_STEP};
}
