//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/profiling/common.hpp"
#include "vpux/utils/profiling/parser/hw.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "schema/graphfile_generated.h"

namespace vpux::profiling {

class RawProfilingRecord;
using RawProfilingRecordPtr = std::shared_ptr<RawProfilingRecord>;
using RawProfilingRecords = std::vector<RawProfilingRecordPtr>;

using WorkpointRecords = std::vector<std::pair<WorkpointConfiguration_t, size_t>>;

enum class SynchronizationPointKind { DMA_TO_DPU, DPU_TO_DMA, DMA_TO_UPA, STRICT_DMA_TO_DPU };

// Container for conjucted storage of tasks of one format: RawProfilingRecordPtr/TaskInfo
struct RawProfilingData {
    RawProfilingRecords dmaTasks;
    RawProfilingRecords dpuTasks;
    RawProfilingRecords swTasks;
    RawProfilingRecords m2iTasks;
    // Pair of workpoint and offset
    WorkpointRecords workpoints;
    // Vector of [ExecutorType; offset in blob(bytes)]
    std::vector<std::pair<ExecutorType, size_t>> parseOrder;
};

// Map of exec. type to section offset and size
using RawDataLayout = std::map<ExecutorType, std::pair<uint32_t, uint32_t>>;

struct RawData {
    RawDataLayout sections;
    RawProfilingData rawRecords;
    MVCNN::TargetDevice device;
    std::optional<double> maybe30XXNceFreq;
};

/**
 * @fn getRawProfilingTasks
 * @brief Show raw counters for debug purpose. Intended for use in prof_parser only
 * @param blobData pointer to the buffer with blob binary
 * @param blobSize blob size in bytes
 * @param profData pointer to the buffer with raw profiling data
 * @param profSize raw profiling data size
 * @param ignoreSanitizationErrors to ignore sanitization errors
 * @return RawProfilingData
 */
RawData getRawProfilingTasks(const uint8_t* blobData, size_t blobSize, const uint8_t* profData, size_t profSize,
                             bool ignoreSanitizationErrors = false);

struct FrequenciesSetup {
public:
    static constexpr double UNITIALIZED_FREQUENCY_VALUE = -1;
    static constexpr double MIN_FREQ_MHZ = 700.0;

public:
    double vpuClk = UNITIALIZED_FREQUENCY_VALUE;
    double dpuClk = UNITIALIZED_FREQUENCY_VALUE;
    double profClk = UNITIALIZED_FREQUENCY_VALUE;
    double dmaBandwidth = UNITIALIZED_FREQUENCY_VALUE;
    bool hasSharedDmaSwCounter = false;
    bool hasSharedDmaDpuCounter = false;
};

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

// freq.cpp

FrequenciesSetup getFrequencySetup(const MVCNN::TargetDevice device, const WorkpointRecords& workpoints,
                                   const std::optional<double>& maybe30XXNceFreq, bool highFreqPerfClk, bool fpga,
                                   vpux::Logger& log);

// sync.cpp

std::optional<int64_t> getDMA2OtherTimersShift(const RawProfilingRecords& dmaTasks,
                                               const RawProfilingRecords& otherTasks, FrequenciesSetup frequenciesSetup,
                                               SynchronizationPointKind pointKind, vpux::Logger& log);

}  // namespace vpux::profiling
