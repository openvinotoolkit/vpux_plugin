//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/profiling/taskinfo.hpp"

#include <ostream>
#include <vector>

namespace vpux::profiling {

void printProfilingAsText(const std::vector<TaskInfo>& tasks, const std::vector<LayerInfo>& layers,
                          std::ostream& output);
void printProfilingAsTraceEvent(const std::vector<TaskInfo>& tasks, const std::vector<LayerInfo>& layers,
                                FreqInfo dpuFreq, std::ostream& output, Logger& log = Logger::global());

//
//  Run profiling post-processing and profilng environemnt hooks
//
std::vector<LayerInfo> getLayerProfilingInfoHook(const uint8_t* profData, size_t profSize, const uint8_t* blobData,
                                                 size_t blobSize);
std::vector<LayerInfo> getLayerProfilingInfoHook(const std::vector<uint8_t>& data, const std::vector<uint8_t>& blob);
std::vector<LayerInfo> getLayerProfilingInfoHook(const uint8_t* profData, size_t profSize,
                                                 const std::vector<uint8_t>& blob);

}  // namespace vpux::profiling
