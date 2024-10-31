//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/core/memory_usage.hpp"

#include <fstream>
#include <regex>
#include <sstream>

namespace vpux {

vpux::KB getPeakMemoryUsage() {
    size_t peakMemUsageKB = 0;

    std::ifstream statusFile("/proc/self/status");
    std::string line;
    std::regex vmPeakRegex("VmPeak:");
    std::smatch vmMatch;
    while (std::getline(statusFile, line)) {
        if (std::regex_search(line, vmMatch, vmPeakRegex)) {
            std::istringstream iss(vmMatch.suffix());
            iss >> peakMemUsageKB;
        }
    }
    return vpux::KB(static_cast<int64_t>(peakMemUsageKB));
}

}  // namespace vpux
