//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/minimal_vf_scheduling.hpp"

using namespace vpux;
using namespace VPU;

MinimalRequirementsVFScheduling::MinimalRequirementsVFScheduling(Logger log, bool prefetching)
        : VFScheduling(log, prefetching) {
}

bool MinimalRequirementsVFScheduling::validate(VFConfig& /*config*/, const TilingOperationStorage::UPtr& tilingInfo,
                                               const Byte /*reservedMemory*/) const {
    return tilingInfo != nullptr;
}

void MinimalRequirementsVFScheduling::correctInputPrefetchingCost(
        StrategyCost& prefetchCost, mlir::Operation* operation, VFConfig& config,
        const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost, const size_t /*index*/) const {
    const auto isInput = llvm::find(config.getInputs(), operation) != config.getInputs().end();

    if (isInput) {
        return;
    }

    StrategyCost parentCost = getParentCost(operation, isolatedOperCost);

    prefetchCost = parentCost <= prefetchCost ? prefetchCost - parentCost : 0;
}

VFScenario MinimalRequirementsVFScheduling::getType() const {
    return VFScenario::MINIMAL;
}
