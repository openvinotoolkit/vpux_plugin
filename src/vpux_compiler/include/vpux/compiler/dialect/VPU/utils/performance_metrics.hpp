//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <cstdint>
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

namespace vpux {
namespace VPU {

constexpr double INVALID_AF = -1;

uint32_t getFreqBase(VPU::ArchKind arch);
uint32_t getFreqStep(VPU::ArchKind arch);
uint32_t getBWBase();
uint32_t getBWStep();
uint32_t getNumEntries();
const SmallVector<float>& getBWScales();
SmallVector<SmallVector<uint64_t>> getBWTicks(mlir::ModuleOp module);
double getActivityFactor(VPU::ExecutorKind execKind, mlir::ModuleOp module, IERT::ComputeResourceOpInterface res);

}  // namespace VPU
}  // namespace vpux
