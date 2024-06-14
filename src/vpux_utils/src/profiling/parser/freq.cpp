//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/profiling/parser/parser.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/logger.hpp"

using namespace vpux::profiling;

namespace {

constexpr double Dma20Bandwidth = 700. / 20000.;
constexpr double Dma27Bandwidth = 1300. / 31200.;
constexpr double Dma40Bandwidth = 1700. / 45000.;

struct Dma27BandwidthProvider {
    static constexpr double value = Dma27Bandwidth;
};

struct Dma40BandwidthProvider {
    static constexpr double value = Dma40Bandwidth;
};

template <typename BandwidthProvider, bool SHARED_DMA_SW_CNT, bool SHARED_DMA_DPU_CNT>
constexpr FrequenciesSetup getFreqSetupHelper(const double vpuFreq, const double dpuFreq, double profClk) {
    return FrequenciesSetup{vpuFreq, dpuFreq, profClk, BandwidthProvider::value, SHARED_DMA_SW_CNT, SHARED_DMA_DPU_CNT};
}

constexpr auto getFreqSetup37XXHelper = getFreqSetupHelper<Dma27BandwidthProvider, true, false>;
constexpr auto getFreqSetup40XXHelper = getFreqSetupHelper<Dma40BandwidthProvider, true, true>;

uint16_t getPllValueChecked(const WorkpointRecords& workpoints, vpux::Logger& log) {
    VPUX_THROW_WHEN(workpoints.empty(), "Expected workpoint data");
    // PLL value from begin of inference
    const auto pllMultFirst = workpoints.front().first.pllMultiplier;
    // PLL value from end of inference
    const auto pllMultLast = workpoints.back().first.pllMultiplier;
    if (pllMultFirst != pllMultLast) {
        log.warning("Frequency changed during the inference: {0} != {1}", pllMultFirst, pllMultLast);
    }
    return pllMultFirst;
}

FrequenciesSetup get37XXSetup(uint16_t pllMult) {
    if (pllMult < 10 || pllMult > 42) {
        vpux::Logger::global().warning("PLL multiplier '{0}' is out of [10; 42] range. MAX freq. setup will be used.",
                                       pllMult);
        return getFreqSetup37XXHelper(975.0, 1300.0, ProfClk37XX::PROF_CLK_DEFAULT_VALUE_MHZ);
    }
    const double VPU_TO_PLL_RATIO = 25.0;
    const double DPU_TO_VPU_RATIO = 4.0 / 3.0;

    const double vpuFreq = pllMult * VPU_TO_PLL_RATIO;
    const double dpuFreq = vpuFreq * DPU_TO_VPU_RATIO;
    return getFreqSetup37XXHelper(vpuFreq, dpuFreq, ProfClk37XX::PROF_CLK_DEFAULT_VALUE_MHZ);
}

FrequenciesSetup get40XXSetup(uint16_t pllMult, bool highFreqPerfClk) {
    if (pllMult < 8 || pllMult > 78) {
        vpux::Logger::global().warning("PLL multiplier '{0}' is out of [8; 78] range. MAX freq. setup will be used.",
                                       pllMult);
        return highFreqPerfClk ? getFreqSetup40XXHelper(1057.14, 1850.0, ProfClk40XX::PROF_CLK_HIGHFREQ_VALUE_MHZ)
                               : getFreqSetup40XXHelper(1057.14, 1850.0, ProfClk40XX::PROF_CLK_DEFAULT_VALUE_MHZ);
    }

    const double DPU_TO_PLL_RATIO = 25.0;
    const double VPU_TO_DPU_RATIO = 4.0 / 7.0;

    const double dpuFreq = pllMult * DPU_TO_PLL_RATIO;
    const double vpuFreq = dpuFreq * VPU_TO_DPU_RATIO;
    return getFreqSetup40XXHelper(
            vpuFreq, dpuFreq,
            highFreqPerfClk ? ProfClk40XX::PROF_CLK_HIGHFREQ_VALUE_MHZ : ProfClk40XX::PROF_CLK_DEFAULT_VALUE_MHZ);
}

FrequenciesSetup getFpgaFreqSetup(MVCNN::TargetDevice device) {
    switch (device) {
    case MVCNN::TargetDevice::TargetDevice_VPUX40XX:
        return getFreqSetup40XXHelper(2.86, 5.0, 1.176);
    default:
        VPUX_THROW("TargetDevice {0} is not supported ", MVCNN::EnumNameTargetDevice(device));
    }
}

FrequenciesSetup getFreqSetupFromPll(MVCNN::TargetDevice device, uint16_t pll, bool highFreqPerfClk) {
    switch (device) {
    case MVCNN::TargetDevice::TargetDevice_VPUX37XX:
        return get37XXSetup(pll);
    case MVCNN::TargetDevice::TargetDevice_VPUX40XX:
        return get40XXSetup(pll, highFreqPerfClk);
    default:
        VPUX_THROW("TargetDevice {0} is not supported ", MVCNN::EnumNameTargetDevice(device));
    }
}

}  // namespace

FrequenciesSetup vpux::profiling::getFrequencySetup(const MVCNN::TargetDevice device,
                                                    const WorkpointRecords& workpoints,
                                                    const std::optional<double>& maybe30XXNceFreq, bool highFreqPerfClk,
                                                    bool fpga, vpux::Logger& log) {
    const bool isDeviceSupportsHighPerfClk = device == MVCNN::TargetDevice::TargetDevice_VPUX40XX;
    if (!isDeviceSupportsHighPerfClk && highFreqPerfClk) {
        log.warning("Requested perf_clk high frequency value is not supported on this device. Default value for the "
                    "device will be used for frequency setup.");
    }
    const bool isPllCaptureSupported = device == MVCNN::TargetDevice::TargetDevice_VPUX37XX ||
                                       device == MVCNN::TargetDevice::TargetDevice_VPUX40XX;
    FrequenciesSetup frequenciesSetup;

    if (isPllCaptureSupported) {
        uint16_t pllMult = 0;
        if (workpoints.size()) {
            pllMult = getPllValueChecked(workpoints, log);
        } else {
            log.warning("No frequency data");
        }
        log.trace("Got PLL value '{0}'", pllMult);

        frequenciesSetup = fpga ? getFpgaFreqSetup(device) : getFreqSetupFromPll(device, pllMult, highFreqPerfClk);
    } else if (device == MVCNN::TargetDevice::TargetDevice_VPUX30XX) {
        VPUX_THROW_UNLESS(maybe30XXNceFreq.has_value(), "No frequency data");
        frequenciesSetup.profClk = maybe30XXNceFreq.value();
        frequenciesSetup.dmaBandwidth = Dma20Bandwidth;
    } else {
        VPUX_THROW("TargetDevice {0} is not supported ", MVCNN::EnumNameTargetDevice(device));
    }
    log.trace("Frequency setup is profClk={0}MHz, vpuClk={1}MHz, dpuClk={2}MHz", frequenciesSetup.profClk,
              frequenciesSetup.vpuClk, frequenciesSetup.dpuClk);

    return frequenciesSetup;
}
