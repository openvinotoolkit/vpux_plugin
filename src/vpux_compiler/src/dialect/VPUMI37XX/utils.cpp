//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI37XX/utils.hpp"

namespace vpux {
namespace VPUMI37XX {

std::pair<uint8_t, uint32_t> getMaxVID(mlir::Operation::operand_range range) {
    uint32_t maxBarrierVID = 0;
    uint8_t maxBarrierVidPid = 0;

    for (auto bar : range) {
        if (auto waitBar = mlir::dyn_cast<VPUMI37XX::ConfigureBarrierOp>(bar.getDefiningOp())) {
            auto waitBarVID = waitBar.getType().getValue();
            if (waitBarVID > maxBarrierVID) {
                maxBarrierVID = waitBarVID;
                maxBarrierVidPid = waitBar.getId();
            }
        }
    }

    return {maxBarrierVidPid, maxBarrierVID};
}

uint64_t computeMask(mlir::Operation::operand_range barriers) {
    uint64_t mask = 0;
    for (auto barrier : barriers) {
        mask |= static_cast<uint64_t>(1) << mlir::cast<VPUMI37XX::ConfigureBarrierOp>(barrier.getDefiningOp()).getId();
    }
    return mask;
}

bool isSwKernelCacheOp(VPUMI37XX::ActKernelRangeOp kernelRange) {
    auto kernelTaskType = kernelRange.getKernelTaskType();
    auto taskType = VPU::symbolizeActShaveTaskType(kernelTaskType.getLeafReference().strref());
    VPUX_THROW_UNLESS(taskType.has_value(), "Operation '{0}' has invalid task type attribute '{1}'", kernelRange,
                      kernelTaskType.getLeafReference());
    return taskType.value() != VPU::ActShaveTaskType::COMPUTE;
}

}  // namespace VPUMI37XX
}  // namespace vpux
