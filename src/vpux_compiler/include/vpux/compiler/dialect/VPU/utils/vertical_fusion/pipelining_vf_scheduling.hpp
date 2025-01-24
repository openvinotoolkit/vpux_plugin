//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_scheduler_interface.hpp"

namespace vpux {
namespace VPU {

/*
  Scheduling scenario for pipelining dpu and act shave operations
*/
class PipeliningVFScheduling : public VFScheduling, public IVFPipelinedScheduling {
public:
    PipeliningVFScheduling(Logger log, bool prefetching);

    /*
      Validate memory requirements
    */
    bool validate(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                  const Byte reservedMemory = Byte(0)) const override;

    /*
      Calculate VF cost
    */
    StrategyCost getCost(VFConfig& config, int64_t tilesNumber, const TilingOperationStorage::UPtr& tilingInfo,
                         const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const override;

    /*
      Type of scenario
    */
    VFScenario getType() const override;

    /*
      Get pipelining
    */
    VFPipelineContainer getPipelining(VFConfig& config, int64_t tilesNumber,
                                      const TilingOperationStorage::UPtr& tilingInfo,
                                      const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const override;

private:
    /*
      Add spill of output dmas ti pipelining
    */
    void addOutputSpill(VFConfig& config, mlir::Operation* operation, VFPipelineContainer& pipelinedStructure,
                        int64_t index, const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                        const VPUNNCostParameters& costParameters) const;
};

}  // namespace VPU
}  // namespace vpux
