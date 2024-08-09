//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/factories/split_cost_getter.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/impl/split_cost_getter.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

VPUIP::SplitCostCb VPUIP::getSplitCostCb(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::NPU37XX:
    case VPU::ArchKind::NPU40XX: {
        return VPUIP::arch37xx::computeSplitCost;
    }
    case VPU::ArchKind::UNKNOWN:
    default: {
        VPUX_THROW("Unexpected architecture {0}", arch);
    }
    }
}
