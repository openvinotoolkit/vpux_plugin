//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/utils/profiling/parser/freq.hpp"
#include "vpux/utils/profiling/parser/parser.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/logger.hpp"

#include <utility>

using namespace vpux::profiling;

namespace {

template <bool SHARED_DMA_SW_CNT, bool SHARED_DMA_DPU_CNT>
constexpr FrequenciesSetup getFreqSetupHelper(const double vpuFreq, const double dpuFreq, double profClk) {
    return FrequenciesSetup{vpuFreq, dpuFreq, profClk, SHARED_DMA_SW_CNT, SHARED_DMA_DPU_CNT, FreqStatus::UNKNOWN};
}

constexpr auto getFreqSetup37XXHelper = getFreqSetupHelper<true, false>;
constexpr auto getFreqSetup40XXHelper = getFreqSetupHelper<true, true>;

std::pair<uint16_t, FreqStatus> getPllValueChecked(const WorkpointRecords& workpoints) {
    if (workpoints.empty()) {
        return {0, FreqStatus::UNKNOWN};
    }
    // PLL value from the beginning of inference
    const auto pllMultFirst = workpoints.front().first.pllMultiplier;
    // PLL value from the end of inference
    const auto pllMultLast = workpoints.back().first.pllMultiplier;

    return {pllMultFirst, (pllMultFirst == pllMultLast) ? FreqStatus::VALID : FreqStatus::INVALID};
}

FrequenciesSetup get37XXSetup(uint16_t pllMult) {
    if (pllMult < 10 || pllMult > 48) {
        vpux::Logger::global().warning("PLL multiplier '{0}' is out of [10; 42] range. MAX freq. setup will be used.",
                                       pllMult);
        pllMult = 39;  // 975 / 1300 MHz
    }
    const double base = 50.0 * pllMult;
    const double vpuFreq = base / 2.0;
    const double dpuFreq = base / 1.5;
    return getFreqSetup37XXHelper(vpuFreq, dpuFreq, ProfClk37XX::PROF_CLK_DEFAULT_VALUE_MHZ);
}

FrequenciesSetup get40XXSetup(uint16_t pllMult, bool highFreqPerfClk) {
    if (pllMult < 8 || pllMult > 78) {
        vpux::Logger::global().warning("PLL multiplier '{0}' is out of [8; 78] range. MAX freq. setup will be used.",
                                       pllMult);
        pllMult = 74;  // 1057 / 1850 MHz
    }
    const double base = 50.0 * pllMult;
    const double vpuFreq = base / 3.5;
    const double dpuFreq = base / 2.0;
    return getFreqSetup40XXHelper(
            vpuFreq, dpuFreq,
            highFreqPerfClk ? ProfClk40XX::PROF_CLK_HIGHFREQ_VALUE_MHZ : ProfClk40XX::PROF_CLK_DEFAULT_VALUE_MHZ);
}

FrequenciesSetup getFpgaFreqSetup(TargetDevice device) {
    switch (device) {
    case TargetDevice::TargetDevice_VPUX40XX:
        return getFreqSetup40XXHelper(2.86, 5.0, 1.176);
    default:
        VPUX_THROW("TargetDevice {0} is not supported ", EnumNameTargetDevice(device));
    }
}

FrequenciesSetup getFreqSetupFromPll(TargetDevice device, uint16_t pll, bool highFreqPerfClk) {
    switch (device) {
    case TargetDevice::TargetDevice_VPUX37XX:
        VPUX_THROW_WHEN(highFreqPerfClk, "Requested perf_clk high frequency value is not supported on this device.");
        return get37XXSetup(pll);
    case TargetDevice::TargetDevice_VPUX40XX:
        return get40XXSetup(pll, highFreqPerfClk);
    default:
        VPUX_THROW("TargetDevice {0} is not supported ", EnumNameTargetDevice(device));
    }
}

}  // namespace

FrequenciesSetup vpux::profiling::getFrequencySetup(const TargetDevice device, const WorkpointRecords& workpoints,
                                                    bool highFreqPerfClk, bool fpga, vpux::Logger& log) {
    FrequenciesSetup frequenciesSetup;

    if (fpga) {
        frequenciesSetup = getFpgaFreqSetup(device);
    } else {
        auto [pllMult, freqStatus] = getPllValueChecked(workpoints);
        if (freqStatus == FreqStatus::UNKNOWN) {
            log.warning("No frequency data");
        } else {
            log.trace("Got PLL value '{0}'", pllMult);
        }
        if (freqStatus == FreqStatus::INVALID) {
            log.warning("Frequency changed during the inference!");
        }

        frequenciesSetup = getFreqSetupFromPll(device, pllMult, highFreqPerfClk);
        frequenciesSetup.clockStatus = freqStatus;
    }
    log.trace("Frequency setup is profClk={0}MHz, vpuClk={1}MHz, dpuClk={2}MHz", frequenciesSetup.profClk,
              frequenciesSetup.vpuClk, frequenciesSetup.dpuClk);

    return frequenciesSetup;
}
