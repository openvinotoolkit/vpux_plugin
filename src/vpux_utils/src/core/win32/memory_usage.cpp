//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/core/memory_usage.hpp"

#include <windows.h>

#include <psapi.h>

namespace vpux {

vpux::KB getPeakMemoryUsage() {
    PROCESS_MEMORY_COUNTERS memCounters;
    GetProcessMemoryInfo(GetCurrentProcess(), &memCounters, sizeof(memCounters));
    return vpux::KB(vpux::Byte(memCounters.PeakWorkingSetSize));
}

}  // namespace vpux
