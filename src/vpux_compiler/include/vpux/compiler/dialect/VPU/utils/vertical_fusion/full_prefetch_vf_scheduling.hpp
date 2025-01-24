//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/weights_prefetch_vf_scheduling.hpp"

namespace vpux {
namespace VPU {

/*
  Scheduling scenario with full prefetching input and output dmas
*/
class FullPrefetchingVFScheduling : public WeightsPrefetchingVFScheduling {
public:
    FullPrefetchingVFScheduling(Logger log, bool prefetching);

    /*
      Validate memory requirements
    */
    bool validate(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                  const Byte reservedMemory = Byte(0)) const override;

    /*
      Type of scenario
    */
    VFScenario getType() const override;

protected:
    void correctOutputSpillCost(StrategyCost& spillCost, VFConfig& config,
                                const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost, const int64_t index,
                                const int64_t tilesNumber) const override;
};

}  // namespace VPU
}  // namespace vpux
