//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace vpux::profiling {

// This structure describes a single entry in Tracing Event format.
struct TraceEventDesc {
    std::string name;
    std::string category;
    int pid;
    int tid;
    // Trace Event Format expects timestamp and duration in microseconds but it may be fractional
    // JSON numbers are doubles anyways so use double for more flexibility
    double timestamp;
    double duration;
    std::vector<std::pair<std::string, std::string>> customArgs;
};

// This stream operator prints TraceEventDesc in JSON format.
// Support Tracing Event Format's "X" event type only - so called "Complete event".
std::ostream& operator<<(std::ostream& os, const TraceEventDesc& event);

}  // namespace vpux::profiling
