//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// Profiling parser public interface

#pragma once

#include "vpux/utils/profiling/taskinfo.hpp"

#include <cstddef>
#include <ostream>
#include <vector>

namespace vpux::profiling {

/**
 * @enum VerbosityLevel
 * @brief Declares verbosity level of printing information
 */
enum VerbosityLevel {
    LOW = 0,     ///< Default, only DMA/SW/Aggregated DPU info
    MEDIUM = 1,  ///< Extend by cluster level information
    HIGH = 5,    ///< Full information including individual variants timings
};

/**
 * @fn getTaskInfo
 * @brief Parse raw profiling output to get per-tasks info.
 * @param blobData pointer to the buffer with blob binary
 * @param blobSize blob size in bytes
 * @param profData pointer to the buffer with raw profiling data
 * @param profSize raw profiling data size
 * @param verbosity amount of DPU info to print, may be LOW|MEDIUM|HIGH
 * @param fpga whether buffer was obtained from FPGA
 * @param highFreqPerfClk use the high frequency perf_clk value (NPU40XX only)
 * @see TaskType
 * @see VerbosityLevel
 * @return std::vector of TaskInfo structures
 */
std::vector<TaskInfo> getTaskInfo(const uint8_t* blobData, size_t blobSize, const uint8_t* profData, size_t profSize,
                                  VerbosityLevel verbosity, bool fpga = false, bool highFreqPerfClk = false);

/**
 * @fn getLayerInfo
 * @brief Parse raw profiling output to get per-layer info.
 * @param blobData pointer to the buffer with blob binary
 * @param blobSize blob size in bytes
 * @param profData pointer to the buffer with raw profiling data
 * @param profSize raw profiling data size
 * @param fpga whether buffer was obtained from FPGA
 * @param highFreqPerfClk use the high frequency perf_clk value (NPU40XX only)
 * @return std::vector of LayerInfo structures
 */
std::vector<LayerInfo> getLayerInfo(const uint8_t* blobData, size_t blobSize, const uint8_t* profData, size_t profSize,
                                    bool fpga = false, bool highFreqPerfClk = false);

/**
 * @fn getLayerInfo
 * @brief Parse raw profiling output to get per-layer info. Reuses precomputed info about tasks.
 * @param taskInfo output from \b getTaskInfo function.
 * @return std::vector of LayerInfo structures
 * @see getTaskInfo
 */
std::vector<LayerInfo> getLayerInfo(const std::vector<TaskInfo>& taskInfo);

void writeDebugProfilingInfo(std::ostream& outStream, const uint8_t* blobData, size_t blobSize, const uint8_t* profData,
                             size_t profSize);

}  // namespace vpux::profiling
