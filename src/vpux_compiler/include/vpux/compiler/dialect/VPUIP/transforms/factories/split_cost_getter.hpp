//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/interfaces/dpu_tiler.hpp"

namespace vpux::VPUIP {

using SplitCostCb = int64_t (*)(const VPUIP::WorkloadSplit&, const VPUIP::WorkloadCostParams&, VPUNN::VPUCostModel&,
                                LogCb);

SplitCostCb getSplitCostCb(VPU::ArchKind arch);

}  // namespace vpux::VPUIP
