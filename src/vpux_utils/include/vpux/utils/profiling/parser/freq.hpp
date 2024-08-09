//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

namespace vpux::profiling {

struct ProfClk37XX {
    // Default perf_clk value after dividing by the default frequency divider
    static constexpr double PROF_CLK_DEFAULT_VALUE_MHZ = 38.4;
};

struct ProfClk40XX {
    // Default perf_clk value after dividing by the default frequency divider
    static constexpr double PROF_CLK_DEFAULT_VALUE_MHZ = 19.2;
    // High frequency perf_clk value after dividing by the default frequency divider
    static constexpr double PROF_CLK_HIGHFREQ_VALUE_MHZ = 200.0;
};

}  // namespace vpux::profiling
