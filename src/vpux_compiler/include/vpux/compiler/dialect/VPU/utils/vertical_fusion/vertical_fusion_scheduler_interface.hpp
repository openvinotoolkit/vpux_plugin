//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"

namespace vpux {
namespace VPU {

/*
  Interface to provide scheduling scenarios for VF
*/
class IVFScheduling {
public:
    virtual ~IVFScheduling() = default;

    /*
      Validate memory requirements
    */
    virtual bool validate(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                          const Byte reservedMemory = Byte(0)) const = 0;

    /*
      Calculate VF cost
    */
    virtual StrategyCost getCost(VFConfig& config, int64_t tilesNumber, const TilingOperationStorage::UPtr& tilingInfo,
                                 const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const = 0;

    /*
      Type of scenario
    */
    virtual VFScenario getType() const = 0;

    /*
      Dependend checks
    */
    virtual const std::deque<std::shared_ptr<IVFScheduling>>& nextChecks() const = 0;

    /*
      Add dependend check
    */
    virtual void addNext(std::shared_ptr<IVFScheduling> check) = 0;
};

/*
  Interface to provide pipelined
*/
class IVFPipelinedScheduling {
public:
    virtual ~IVFPipelinedScheduling() = default;

    /*
      Get the structure of pipelined operations
    */
    virtual VFPipelineContainer getPipelining(VFConfig& config, int64_t tilesNumber,
                                              const TilingOperationStorage::UPtr& tilingInfo,
                                              const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const = 0;
};

/*
  Base implementation of scheduling scenario features
*/
class VFScheduling : public IVFScheduling {
public:
    VFScheduling(Logger log, bool prefetching = true);
    virtual ~VFScheduling() = default;

    /*
      Calculate VF cost
    */
    StrategyCost getCost(VFConfig& config, int64_t tilesNumber, const TilingOperationStorage::UPtr& tilingInfo,
                         const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const override;

    /*
      Dependend checks
    */
    const std::deque<std::shared_ptr<IVFScheduling>>& nextChecks() const override;

    /*
      Add dependend check
    */
    void addNext(std::shared_ptr<IVFScheduling> check) override;

protected:
    /*
      Calculate input sizes
    */
    Byte getInputsSize(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo) const;

    /*
      Calculate output sizes
    */
    Byte getOutputsSize(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo) const;

    /*
      Get parameters for cost calculation
    */
    VPUNNCostParameters fillInCostParam(mlir::Operation* operation, const OutputTiling& tiling,
                                        const SmallVector<TileInfo>& inputTiles) const;

    /*
      Get parameters for cost calculation
    */
    VPUNNCostParameters fillInCostParam(mlir::Operation* operation, const TilingOperationStorage::UPtr& opStorage,
                                        size_t index) const;

    /*
      Get cost of common case
    */
    StrategyCost getLinearCost(VFConfig& config, int64_t tilesNumber, const TilingOperationStorage::UPtr& tilingInfo,
                               const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const;

    /*
      Prefetch output spill
    */
    virtual void correctOutputSpillCost(StrategyCost& spillCost, VFConfig& config,
                                        const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost,
                                        const int64_t index, const int64_t tilesNumber) const;

    /*
      Prefetch input dmas
    */
    virtual void correctInputPrefetchingCost(StrategyCost& prefetchCost, mlir::Operation* operation, VFConfig& config,
                                             const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost,
                                             const size_t index) const;

    /*
      Get input dmas cost
    */
    StrategyCost getPrefetchingCost(mlir::Operation* operation, VFConfig& config,
                                    const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                    const VPUNNCostParameters& parameters, const bool isInput,
                                    const TilingOperationStorage::UPtr& tilingInfo, const int64_t index) const;

    /*
      Get cost of parent operation
    */
    StrategyCost getParentCost(mlir::Operation* operation,
                               const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost) const;

protected:
    Logger _log;
    bool _prefetching = true;

private:
    std::deque<std::shared_ptr<IVFScheduling>> _dependents;
};

}  // namespace VPU
}  // namespace vpux
