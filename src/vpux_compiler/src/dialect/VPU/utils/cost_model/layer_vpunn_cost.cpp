//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"

using namespace vpux;
using namespace VPU;

MultiClusterStrategySetter::MultiClusterStrategySetter(mlir::Operation* operation, VPU::MultiClusterStrategy strategy)
        : _operation(operation) {
    setTemporaryStrategy(strategy);
}

MultiClusterStrategySetter::~MultiClusterStrategySetter() {
    removeTemporaryStrategy();
}

void MultiClusterStrategySetter::removeTemporaryStrategy() {
    if (auto childClusterOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(_operation)) {
        if (_origStrategy.has_value()) {
            childClusterOp.setMultiClusterStrategy(_origStrategy.value());
        } else {
            _operation->removeAttr(multiClusterStrategy);
        }
    }
}

void MultiClusterStrategySetter::setTemporaryStrategy(VPU::MultiClusterStrategy tempStrategy) {
    if (auto clusterOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(_operation)) {
        _origStrategy = clusterOp.getMultiClusterStrategy();
        clusterOp.setMultiClusterStrategy(tempStrategy);
    }
}

StrategyCost LayerVPUNNCost::getStrategyCost(mlir::Operation* operation, const VPUNNCostParameters& parameters) const {
    if (mlir::isa<VPU::NCEPermuteOp>(operation)) {
        return getSimpleLayerCost(operation, parameters);
    } else if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(operation)) {
        return getNCELayerCost(nceOp, parameters);
    } else if (auto swOp = mlir::dyn_cast<VPU::SWOpInterface>(operation)) {
        return getSWLayerCost(swOp, parameters);
    } else if (VPU::isPureViewOp(operation)) {
        return 0.0;
    } else {
        _log.trace("Unsupported op type {0} at {1}", operation->getName(), operation->getLoc());
        return getSimpleLayerCost(operation, parameters);
    }
}

StrategyCost LayerVPUNNCost::getSpillingWriteCost(mlir::Operation* operation,
                                                  const VPUNNCostParameters& parameters) const {
    StrategyCost writeCost = 0;

    if (!VPU::isPureViewOp(operation)) {
        auto origParentType = operation->getResult(0).getType().cast<vpux::NDTypeInterface>();

        const auto parentTiling =
                parameters._tiling.empty() ? OutputTiling({TileInfo(origParentType.getShape())}) : parameters._tiling;
        writeCost = std::accumulate(
                std::begin(parentTiling), std::end(parentTiling), writeCost, [&](StrategyCost cost, auto& tileInfo) {
                    auto tiledType = origParentType.extractDenseTile(tileInfo.offsets, tileInfo.shape);
                    if (auto parentClusterOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(operation)) {
                        auto numClusters =
                                parentClusterOp.getOptimalNumClusters(origParentType.getShape(), parameters._strategy);

                        tiledType = VPU::getDistributedOutputTypeFromOp(parentClusterOp, tiledType, numClusters,
                                                                        parameters._strategy);
                    }
                    return cost + getDMACost(tiledType, _vpuDevice, _vpunnCostModel, _numDMAPorts);
                });
    }

    return writeCost;
}

StrategyCost LayerVPUNNCost::getSpillingReadCost(mlir::Operation* operation, const VPUNNCostParameters& parameters,
                                                 mlir::Operation* parentOp /*nullptr*/,
                                                 std::function<bool(mlir::Value value)> findOperand /*nullptr*/) const {
    StrategyCost readCost = 0;

    VPUX_THROW_WHEN(parentOp == nullptr && findOperand == nullptr,
                    "Either parent operation or functor for operands must be passed");

    if (!VPU::isPureViewOp(operation)) {
        if (findOperand == nullptr) {
            findOperand = [&](auto value) {
                auto operation = value.getDefiningOp();
                while (operation != nullptr) {
                    if (operation == parentOp) {
                        return true;
                    } else if (VPU::isPureViewOp(operation)) {
                        operation = operation->getOperand(0).getDefiningOp();
                        continue;
                    }
                    return false;
                }
                return false;
            };
        }

        const auto operandItr = llvm::find_if(operation->getOperands(), findOperand);
        VPUX_THROW_WHEN(operandItr == operation->getOperands().end(),
                        "Operation {0} has no common tensors with operation {1}", *parentOp, *operation);
        const size_t operandInd = std::distance(operation->getOperands().begin(), operandItr);
        const auto childTiling = parameters._tiling.empty()
                                         ? OutputTiling({TileInfo(getShape(operation->getResult(0)))})
                                         : parameters._tiling;

        MultiClusterStrategySetter mcSetter(operation, parameters._strategy);

        readCost = std::accumulate(
                std::begin(childTiling), std::end(childTiling), readCost, [&](StrategyCost cost, auto& tileInfo) {
                    const auto childOperandsTiling = getTileTypes(operation, tileInfo);
                    VPUX_THROW_WHEN(childOperandsTiling.size() <= operandInd,
                                    "Incorrect number of types {0} for operands of operation {1}",
                                    childOperandsTiling.size(), operation->getLoc());

                    return cost +
                           getDMACost(childOperandsTiling[operandInd], _vpuDevice, _vpunnCostModel, _numDMAPorts);
                });
    }

    return readCost;
}

StrategyCost LayerVPUNNCost::getSpillingCost(mlir::Operation* parentOp, const VPUNNCostParameters& parentParameters,
                                             mlir::Operation* childOp,
                                             const VPUNNCostParameters& childParameters) const {
    /*
     Spilling cost is computed as sum of cyclecost of dma of parent operation from CMX->DDR (write cost)
     and cyclecost of dma of child operation from DDR->CMX (read cost)
     In case one of operations is pure view like, it's supposed to be in DDR already, no write/read cost
     is needed from/to it
    */

    return getSpillingWriteCost(parentOp, parentParameters) + getSpillingReadCost(childOp, childParameters, parentOp);
}

size_t LayerVPUNNCost::getNumClusterCorrectionSize(VPU::MultiClusterStrategy strategy) const {
    return strategy != MultiClusterStrategy::Clustering ? _numTiles : 1;
}

StrategyCost LayerVPUNNCost::getSimpleLayerCost(mlir::Operation* operation,
                                                const VPUNNCostParameters& parameters) const {
    auto outputType = operation->getResult(0).getType().cast<vpux::NDTypeInterface>();

    return outputType.getTotalAllocSize().count() / getNumClusterCorrectionSize(parameters._strategy);
}

StrategyCost LayerVPUNNCost::getNCELayerCost(VPU::NCEOpInterface nceOp, const VPUNNCostParameters& parameters) const {
    // Types for each tile
    SmallVector<SmallVector<NDTypeInterface>> tilesTypes;

    auto isPrefetchTilingEnabled = (parameters._mode != TilingMode::ISOLATED);

    _log.trace("Start calculating vpunn cost for Op {0} with strategy {1}", nceOp.getLoc(), parameters._strategy);

    const auto costParams = VPU::getWorkloadCostParam(nceOp, _arch, _numDPUs);
    MultiClusterStrategySetter mcSetter(nceOp, parameters._strategy);
    // Set prefetching to be true to ignore the DMA cost and only get the execution DPU cost
    // According to the VPUNN API definition,
    //      when prefetching is false, the returned cost is the sum of DPU + weights DMA
    //      when prefetching is true, the returned cost is just DPU because it considers the weights are prefetched
    const auto vpunnStrategy = VPU::getVPULayerStrategy(parameters._strategy, _numDPUs, _numTiles, _numShaveActs, true);
    auto vpunnLayerDPUCosts = getDPUCostForNCEOp(nceOp, parameters._strategy, parameters._tiling, tilesTypes,
                                                 costParams, vpunnStrategy, _vpunnCostModel, _log);
    _log.trace("VPUNN DPU layer costs {0}", vpunnLayerDPUCosts);

    if (vpunnLayerDPUCosts.empty()) {
        _log.trace("DPU cost is empty, return COST_MAX");
        return std::numeric_limits<VPU::StrategyCost>::max();
    }

    // Accumulate DPU costs
    auto vpunnCost = std::accumulate(vpunnLayerDPUCosts.begin(), vpunnLayerDPUCosts.end(), 0);

    // Add extra weights DMA costs
    const auto getSpillingReadCost = [&](NDTypeInterface srcType) -> uint32_t {
        return checked_cast<uint32_t>(getDMACost(srcType, _vpuDevice, _vpunnCostModel, _numDMAPorts));
    };
    auto vpunnLayerWeightsCosts = getPerTileWeightsDMACosts(nceOp, tilesTypes, getSpillingReadCost);
    _log.trace("VPUNN weights DMA costs {0}", vpunnLayerWeightsCosts);
    vpunnCost += getWeightsDMACostForNCEOp(nceOp, parameters._tiling, vpunnLayerDPUCosts, vpunnLayerWeightsCosts,
                                           isPrefetchTilingEnabled, _log);
    _log.trace("VPUNN total layer cost {0}", vpunnCost);
    return vpunnCost;
}

StrategyCost LayerVPUNNCost::getSWLayerCost(VPU::SWOpInterface swOp, const VPUNNCostParameters& parameters) const {
    auto outputType = swOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto outputTiling = parameters._tiling;
    if (outputTiling.empty()) {
        outputTiling.push_back(TileInfo(outputType.getShape()));
    }

    StrategyCost fullCost = 0;
    for (auto index : irange(outputTiling.size())) {
        SmallVector<vpux::NDTypeInterface> inputNDTypes;
        SmallVector<TileInfo> inputTiles;
        if (parameters._operandsTiling.empty()) {
            if (auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(swOp.getOperation())) {
                inputTiles = tilingBuilderOp.backInferTileInfo(outputTiling[index], _log).tiles;
            } else {
                continue;
            }
        } else {
            inputTiles = parameters._operandsTiling[index];
        }

        for (auto typeIndex : irange(inputTiles.size())) {
            inputNDTypes.push_back(swOp->getOperand(typeIndex).getType().cast<vpux::NDTypeInterface>().extractDenseTile(
                    inputTiles[typeIndex].offsets, inputTiles[typeIndex].shape));
        }
        const auto vpunnLayer = getVPUNNSWKernelOp(
                swOp, outputType.extractDenseTile(outputTiling[index].offsets, outputTiling[index].shape),
                inputNDTypes);
        if (!vpunnLayer) {
            fullCost += getSimpleLayerCost(swOp, parameters);
        } else {
            auto vpunnStrategy =
                    VPU::getVPULayerStrategy(parameters._strategy, _numDPUs, _numTiles, _numShaveActs, false);
            fullCost += _vpunnCostModel->Layer(*vpunnLayer, vpunnStrategy);
        }
    }

    return fullCost;
}
