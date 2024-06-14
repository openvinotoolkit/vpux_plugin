//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "intel_npu/al/config/common.hpp"
#include "intel_npu/al/config/compiler.hpp"

#include "vpux/compiler/compiler.hpp"
#include "vpux/compiler/options_mapper.hpp"
#include "vpux/utils/IE/private_properties.hpp"

#include "vpux/compiler/NPU37XX/pipelines.hpp"
#include "vpux/compiler/NPU40XX/pipelines.hpp"
#include "vpux/compiler/VPU30XX/pipelines.hpp"

#include <device_helpers.hpp>

using namespace vpux;

namespace {

std::optional<int> getMaxTilesValue(const Config& config) {
    if (config.has<intel_npu::MAX_TILES>()) {
        auto logger = vpux::Logger::global();
        int maxTiles = checked_cast<int>(config.get<intel_npu::MAX_TILES>());
        // E#117389: remove overrides and change to exceptions once driver & plugin will be fixed
        const int maxArchTiles = checked_cast<int>(VPU::getMaxArchDPUClusterNum(vpux::getArchKind(config)));
        if (maxTiles < 1 || maxTiles > maxArchTiles) {
            logger.warning("Invalid number of NPU_MAX_TILES for requested arch, got {0}. Override to {1}", maxTiles,
                           maxArchTiles);
            maxTiles = maxArchTiles;
        }
        return maxTiles;
    }
    return std::nullopt;
}

int getMaxDPUClusterNum(const Config& config) {
    const int maxArchTiles = checked_cast<int>(VPU::getMaxArchDPUClusterNum(vpux::getArchKind(config)));
    const auto maybeMaxTiles = getMaxTilesValue(config);
    if (maybeMaxTiles.has_value()) {
        return maybeMaxTiles.value();
    }
    return maxArchTiles;
}

int getNumberOfDPUGroupsUnchecked(const Config& config) {
    const std::string platform = ov::intel_npu::Platform::standardize(config.get<intel_npu::PLATFORM>());

    if (platform == ov::intel_npu::Platform::NPU3720) {
        switch (config.get<intel_npu::PERFORMANCE_HINT>()) {
        case ov::hint::PerformanceMode::THROUGHPUT:
        case ov::hint::PerformanceMode::LATENCY:
        default:
            return getMaxDPUClusterNum(config);
        }
    } else if (platform == ov::intel_npu::Platform::NPU4000) {
        switch (config.get<intel_npu::PERFORMANCE_HINT>()) {
        case ov::hint::PerformanceMode::LATENCY:
            return 4;
        case ov::hint::PerformanceMode::THROUGHPUT:
        default:
            return 2;
        }
    } else {
        switch (config.get<intel_npu::PERFORMANCE_HINT>()) {
        case ov::hint::PerformanceMode::THROUGHPUT:
            return 1;
        case ov::hint::PerformanceMode::LATENCY:
        default:
            return getMaxDPUClusterNum(config);
        }
    }
}

};  // namespace

VPU::InitCompilerOptions vpux::getInitCompilerOptions(const Config& config) {
    const auto archKind = getArchKind(config);
    const auto compilationMode = getCompilationMode(config);
    const auto revisionID = getRevisionID(config);
    const auto numOfDPUGroups = getNumberOfDPUGroups(config);
    const auto numOfDMAPorts = getNumberOfDMAEngines(config);
    const auto wlmRollback = getWlmRollback(config);

    return {archKind, compilationMode, revisionID, numOfDPUGroups, numOfDMAPorts, wlmRollback};
}

//
// getArchKind
//

VPU::ArchKind vpux::getArchKind(const Config& config) {
    const std::string platform = ov::intel_npu::Platform::standardize(config.get<intel_npu::PLATFORM>());

    if (platform == ov::intel_npu::Platform::AUTO_DETECT) {
        return VPU::ArchKind::UNKNOWN;
    } else if (platform == ov::intel_npu::Platform::NPU3700) {
        return VPU::ArchKind::NPU30XX;
    } else if (platform == ov::intel_npu::Platform::NPU3720) {
        return VPU::ArchKind::NPU37XX;
    } else if (platform == ov::intel_npu::Platform::NPU4000) {
        return VPU::ArchKind::NPU40XX;
    } else {
        VPUX_THROW("Unsupported VPUX platform");
    }
}

//
// getCompilationMode
//

VPU::CompilationMode vpux::getCompilationMode(const Config& config) {
    if (!config.has<intel_npu::COMPILATION_MODE>()) {
        return VPU::CompilationMode::DefaultHW;
    }

    const auto parsed = VPU::symbolizeCompilationMode(config.get<intel_npu::COMPILATION_MODE>());
    VPUX_THROW_UNLESS(parsed.has_value(), "Unsupported compilation mode '{0}'",
                      config.get<intel_npu::COMPILATION_MODE>());
    return parsed.value();
}

//
// getRevisionID
//

std::optional<int> vpux::getRevisionID(const Config& config) {
    if (config.has<intel_npu::STEPPING>()) {
        return checked_cast<int>(config.get<intel_npu::STEPPING>());
    }
    return std::nullopt;
}

//
// getNumberOfDPUGroups
//

std::optional<int> vpux::getNumberOfDPUGroups(const Config& config) {
    if (config.has<intel_npu::TILES>() && config.has<intel_npu::DPU_GROUPS>()) {
        VPUX_THROW("Config conflict! NPU_TILES and NPU_DPU_GROUPS both set. Please only set NPU_TILES");
    } else if (config.has<intel_npu::TILES>()) {
        const int requestedNpuTiles = checked_cast<int>(config.get<intel_npu::TILES>());
        VPUX_THROW_WHEN(requestedNpuTiles > getMaxDPUClusterNum(config),
                        "Requested number of NPU tiles is larger than maximum available tiles: {0} > {1}",
                        requestedNpuTiles, getMaxDPUClusterNum(config));
        return requestedNpuTiles;
    } else if (config.has<intel_npu::DPU_GROUPS>()) {
        const int requestedDpuGroups = checked_cast<int>(config.get<intel_npu::DPU_GROUPS>());
        VPUX_THROW_WHEN(requestedDpuGroups > getMaxDPUClusterNum(config),
                        "Requested number of DPU groups is larger than maximum available tiles: {0} > {1}",
                        requestedDpuGroups, getMaxDPUClusterNum(config));
        return requestedDpuGroups;
    }

    const int numOfDpuGroups = getNumberOfDPUGroupsUnchecked(config);
    const auto maybeMaxTiles = getMaxTilesValue(config);
    if (maybeMaxTiles.has_value() && (numOfDpuGroups > maybeMaxTiles.value())) {
        vpux::Logger::global().warning("PERFORMANCE_HINT parameter used more DPU_GROUPS than MAX_TILES");
    }

    return numOfDpuGroups;
}

//
// getNumberOfDMAEngines
//

std::optional<int> vpux::getNumberOfDMAEngines(const Config& config) {
    if (config.has<intel_npu::DMA_ENGINES>()) {
        return checked_cast<int>(config.get<intel_npu::DMA_ENGINES>());
    }

    auto archKind = vpux::getArchKind(config);
    auto numOfDpuGroups = getNumberOfDPUGroups(config);
    int maxDmaPorts = VPU::getMaxDMAPorts(archKind);

    auto getNumOfDmaPortsWithDpuCountLimit = [&]() {
        return std::min(maxDmaPorts, numOfDpuGroups.value_or(maxDmaPorts));
    };

    const std::string platform = ov::intel_npu::Platform::standardize(config.get<intel_npu::PLATFORM>());

    if (platform == ov::intel_npu::Platform::NPU3720) {
        switch (config.get<intel_npu::PERFORMANCE_HINT>()) {
        case ov::hint::PerformanceMode::THROUGHPUT:
        case ov::hint::PerformanceMode::LATENCY:
        default:
            return getNumOfDmaPortsWithDpuCountLimit();
        }
    } else if (platform == ov::intel_npu::Platform::NPU4000) {
        if (!isELFEnabled(config)) {
            // With Graphfile backend only 1 DMA is supported
            return 1;
        }

        switch (config.get<intel_npu::PERFORMANCE_HINT>()) {
        case ov::hint::PerformanceMode::THROUGHPUT:
        case ov::hint::PerformanceMode::LATENCY:
        default:
            return getNumOfDmaPortsWithDpuCountLimit();
        }
    } else {
        switch (config.get<intel_npu::PERFORMANCE_HINT>()) {
        case ov::hint::PerformanceMode::THROUGHPUT:
            return 1;
        case ov::hint::PerformanceMode::LATENCY:
        default:
            return maxDmaPorts;
        }
    }
}

namespace vpux {

template <typename Options>
std::optional<bool> getWlmRollback(const Config& config) {
    const auto options = Options::createFromString(config.get<intel_npu::COMPILATION_MODE_PARAMS>());
    if (options == nullptr) {
        return std::nullopt;
    }

    return options->wlmRollback;
}

template <typename ReferenceSWOptions, typename ReferenceHWOptions, typename DefaultHWOptions>
std::optional<bool> getWlmRollback(const Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        return getWlmRollback<ReferenceSWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::ReferenceHW) {
        return getWlmRollback<ReferenceHWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        return getWlmRollback<DefaultHWOptions>(config);
    } else {
        return std::nullopt;
    }
}

std::optional<bool> getWlmRollback(const Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::NPU30XX) {
        return getWlmRollback<ReferenceSWOptions30XX, ReferenceHWOptions30XX, DefaultHWOptions30XX>(config);
    } else if (arch == VPU::ArchKind::NPU37XX) {
        return getWlmRollback<ReferenceSWOptions37XX, ReferenceHWOptions37XX, DefaultHWOptions37XX>(config);
    } else if (arch == VPU::ArchKind::NPU40XX) {
        return getWlmRollback<ReferenceSWOptions40XX, ReferenceHWOptions40XX, DefaultHWOptions40XX>(config);
    } else {
        return std::nullopt;
    }
}

}  // namespace vpux
