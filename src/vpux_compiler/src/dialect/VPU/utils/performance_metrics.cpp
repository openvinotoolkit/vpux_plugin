//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/performance_metrics.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/frequency_table.hpp"
#include "vpux/utils/profiling/parser/freq.hpp"

namespace vpux {
namespace VPU {

// Base of bandwidth values used in tables (in MB/s).
static constexpr uint32_t BW_BASE = 2000;
// Step of bandwidth values used in tables (in MB/s).
static constexpr uint32_t BW_STEP = 100;
// Num entries in table, each entry contains set of values for particular frequency
static constexpr uint32_t NUM_ENTRIES = 5;
// Profiling timer block fixed frequency by MHz
// For NPU37XX it runs at 38.4 MHz
// TODO: it should be provided by vpunn API per arch
static constexpr double PROF_CLK = 38.4;

uint32_t getBWBase() {
    return BW_BASE;
}
uint32_t getBWStep() {
    return BW_STEP;
}
uint32_t getNumEntries() {
    return NUM_ENTRIES;
}

double getProfClk() {
    return PROF_CLK;
}

const SmallVector<float>& getBWScales() {
    // value in [0.0..1.0] range indicating scalability of network for a given DDR bandwidth.
    static const SmallVector<float> byBWScales({0.0F, 0.2F, 0.4F, 0.6F, 0.8F});
    return byBWScales;
}

SmallVector<SmallVector<uint64_t>> getBWTicks(mlir::ModuleOp module) {
    SmallVector<SmallVector<uint64_t>> ret;
    ret.reserve(VPU::getNumEntries());
    // expected ticks (based on FRC @38.4MHz) an inference should take for a given DDR bandwidth.
    SmallVector<uint64_t> byBWTicks({0UL, 0UL, 0UL, 0UL, 0UL});
    // inferenceTime table will be zero defaultly when InferenceExecutionAnalysisPass disabled
    // In this way the runtime will be able to use measured ticks instead
    size_t inferenceTimebyDPUCycle = 0;
    vpux::IE::CNNNetworkOp netOp;
    mlir::func::FuncOp netFunc;
    vpux::IE::CNNNetworkOp::getFromModule(module, netOp, netFunc);
    if (netOp.getInferenceTiming().has_value()) {
        inferenceTimebyDPUCycle = netOp.getInferenceTiming().value();
    } else {
        vpux::Logger::global().warning("CNNNetworkOp {0} doesn't have value in InferenceTiming attribute. Will use "
                                       "default InferenceTiming and activity factor value instead",
                                       netOp);
    }

    // Get corresponding dpu freq (MHz) from vpunn to parse inferenceTimebyDPUCycle
    const auto arch = VPU::getArch(module);
    size_t dpuBaseFreq = VPU::getDpuFrequency(arch, VPU::getRevisionID(module));
    // Convert inference ticks by getProfClk
    auto profClk = arch == VPU::ArchKind::NPU40XX ? profiling::ProfClk40XX::PROF_CLK_DEFAULT_VALUE_MHZ
                                                  : profiling::ProfClk37XX::PROF_CLK_DEFAULT_VALUE_MHZ;
    auto freqTable = VPU::getFrequencyTable(arch);
    auto freqBase = freqTable().base;
    auto freqStep = freqTable().step;
    size_t baseTicks = static_cast<double>(inferenceTimebyDPUCycle) / static_cast<double>(dpuBaseFreq) * profClk;
    for (uint32_t i = 0; i < VPU::getNumEntries(); ++i) {
        // Scale baseTicks by different dpu freq
        auto dpuFreq = freqBase + i * freqStep;
        auto ticksByDPUFreq = baseTicks * dpuBaseFreq / dpuFreq;
        // Scale ticks by dma bandwidth
        // Currently ignore bandwidth scaling, put same ticks for all bw steps
        for (uint32_t j = 0; j < VPU::getNumEntries(); ++j) {
            byBWTicks[j] = ticksByDPUFreq;
        }
        ret.push_back(SmallVector(byBWTicks));
    }
    return ret;
}

double getActivityFactor(VPU::ExecutorKind execKind, mlir::ModuleOp module, IERT::ComputeResourceOpInterface res) {
    // 0.5 is a recommanded default value for AF by VPUNN team
    double activityFactor = 0.5;
    const auto arch = VPU::getArch(module);
    if (execKind == VPU::ExecutorKind::NCE || execKind == VPU::ExecutorKind::SHAVE_NN) {
        switch (arch) {
        case VPU::ArchKind::NPU37XX:
        case VPU::ArchKind::NPU40XX:
            // Here we must get AF from NCE res (a TileResourceOp) as the AF attribute is attached to tile op
            if (execKind == VPU::ExecutorKind::NCE) {
                auto NCERes = mlir::cast<IE::TileResourceOp>(res.getOperation());
                if (auto factorAttr = NCERes.getActivityFactorAttr()) {
                    activityFactor = factorAttr.getValue().convertToDouble();
                }
            } else {
                auto SHAVERes = mlir::cast<IE::ExecutorResourceOp>(res.getOperation());
                if (auto factorAttr = SHAVERes.getActivityFactorAttr()) {
                    activityFactor = factorAttr.getValue().convertToDouble();
                }
            }
            // In below situation, activityFactor may to be >1
            // 1) when the energy reference is not the maximum powervirus. Eg: the powerVirus for INT is smaller
            // than powerVirus for FLOAT. Now we are using INT8 powervirus (for NPU37XX /w v1.5.9 VPUNN releases) as
            // max power reference so that AF>1 is possible 2) If inferenceTime estimation is smaller than the
            // Energy estimated in powervirusDPUCycles their ratio will be >1. This is transitory because in the
            // real world the measured time will be bigger and the RuntimeNN will normalize the numbers considering
            // real execution time. Eg: NewAF = (OldAF * CompledInferedTimeSmall) / measuredTimeBig. This scenario
            // might happen in some extreme cases.
            if (activityFactor < 0 || activityFactor > 1) {
                vpux::Logger::global().warning("Activity factor value should be in range [0, 1] for general situation "
                                               "but got {0}. Some unexpected cases may happen",
                                               activityFactor);
            }
            return activityFactor;
        default:
            return activityFactor;
        }
    }
    return INVALID_AF;
}

}  // namespace VPU
}  // namespace vpux
