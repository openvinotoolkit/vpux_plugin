//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// Profiling core utils shared between compiler and parser

#pragma once

#include <string>
#include <string_view>

namespace vpux::profiling {

enum class ExecutorType { DPU = 1, UPA = 2, ACTSHAVE = 3, DMA_SW = 4, WORKPOINT = 5, DMA_HW = 6, M2I = 7 };

// DMA HW profiling and workpoint capture require 64B section alignment
constexpr size_t PROFILING_SECTION_ALIGNMENT = 64;
constexpr size_t WORKPOINT_BUFFER_SIZE = 64;  // must be in a separate cache line

constexpr char PROFILING_CMX_2_DDR_OP_NAME[] = "ProfilingCMX2DDR";
constexpr char PROFILING_WORKPOINT_READ_ATTR[] = "PROFWORKPOINT_READ";
constexpr char PROFILING_OUTPUT_NAME[] = "profilingOutput";

std::string convertExecTypeToName(ExecutorType execType);
ExecutorType convertDataInfoNameToExecType(std::string_view name);

}  // namespace vpux::profiling
