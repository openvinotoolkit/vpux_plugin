//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_scheduler_interface.hpp"

namespace vpux {
namespace VPU {

/*
  Scheduling scenario with weights prefetching with last operation
*/
class PrefetchingLastOpVFScheduling : public VFScheduling {
public:
    PrefetchingLastOpVFScheduling(Logger log, bool prefetching);

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
    void correctInputPrefetchingCost(StrategyCost& prefetchCost, mlir::Operation* operation, VFConfig& config,
                                     const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost,
                                     const size_t index) const override;
};

}  // namespace VPU
}  // namespace vpux
