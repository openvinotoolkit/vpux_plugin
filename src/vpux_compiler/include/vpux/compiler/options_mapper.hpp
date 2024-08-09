//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/IE/config.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

namespace vpux {

VPU::InitCompilerOptions getInitCompilerOptions(const intel_npu::Config& config);

VPU::ArchKind getArchKind(const intel_npu::Config& config);
VPU::CompilationMode getCompilationMode(const intel_npu::Config& config);
std::optional<int> getRevisionID(const intel_npu::Config& config);
std::optional<int> getNumberOfDPUGroups(const intel_npu::Config& config);
std::optional<int> getNumberOfDMAEngines(const intel_npu::Config& config);
std::optional<bool> getWlmRollback(const intel_npu::Config& config);
Byte getAvailableCmx(const intel_npu::Config& config);

std::optional<std::string> getPerformanceHintOverride(const intel_npu::Config& config);

}  // namespace vpux
