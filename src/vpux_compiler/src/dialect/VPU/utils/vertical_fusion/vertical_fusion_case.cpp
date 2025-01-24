//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_case.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"

using namespace vpux;
using namespace VPU;

VFCase::~VFCase() {
}

VFCase::VFCase(const VFConfig& config, Dim axis): _config(config), _axis(axis) {
}

VFCase::VFCase(VFCase&& vfCase): _config(vfCase._config) {
    _axis = vfCase._axis;
    _cachedCost = vfCase._cachedCost;
    _vfScheduling = std::move(vfCase._vfScheduling);
    _vfTilingStorage = std::move(vfCase._vfTilingStorage);
    _tilingNumber = vfCase._tilingNumber;

    vfCase._tilingNumber = 1;
    vfCase._vfScheduling = nullptr;
    vfCase._vfTilingStorage = nullptr;
}

VFCase& VFCase::operator=(VFCase&& other) {
    if (this == &other) {
        return *this;
    }

    std::swap(_config, other._config);
    std::swap(_vfScheduling, other._vfScheduling);
    _vfTilingStorage = std::move(other._vfTilingStorage);
    _axis = other._axis;
    _tilingNumber = other._tilingNumber;
    std::swap(_cachedCost, other._cachedCost);

    other._tilingNumber = 1;
    other._vfScheduling = nullptr;
    _vfTilingStorage = nullptr;

    return *this;
}

VFCase::VFCase(const VFCase& vfCase): _config(vfCase._config) {
    _axis = vfCase._axis;
    _cachedCost = vfCase._cachedCost;
    _vfScheduling = vfCase._vfScheduling;
    _tilingNumber = vfCase._tilingNumber;
    _vfTilingStorage = nullptr;
}

VFCase& VFCase::operator=(const VFCase& other) {
    if (this == &other) {
        return *this;
    }

    _config = other._config;
    _axis = other._axis;
    _cachedCost = other._cachedCost;
    _vfScheduling = other._vfScheduling;
    _tilingNumber = other._tilingNumber;
    _vfTilingStorage = nullptr;

    return *this;
}

void VFCase::setTilingNumber(int64_t number) {
    if (_tilingNumber != number) {
        clearCache();
    }
    _tilingNumber = number;
}

void VFCase::setScheduling(std::shared_ptr<IVFScheduling> vfScheduling) {
    if (vfScheduling == nullptr || (_vfScheduling != nullptr && _vfScheduling->getType() != vfScheduling->getType())) {
        clearCache();
    }
    _vfScheduling = std::move(vfScheduling);
}

void VFCase::setTilingStorage(std::unique_ptr<TilingOperationStorage> vfStorage) {
    if (vfStorage == nullptr) {
        clearCache();
    }
    _vfTilingStorage = std::move(vfStorage);
}

void VFCase::clearCache() {
    _cachedCost.reset();
}

void VFCase::addCMXWriteSpills(const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) {
    const auto getStrategy = [](auto* operation) -> VPU::MultiClusterStrategy {
        auto strategy = VPU::MultiClusterStrategy::Clustering;
        if (auto mcOperation = mlir::dyn_cast<VPU::ClusteredOpInterface>(operation)) {
            strategy = mcOperation.getMultiClusterStrategy().value_or(strategy);
        }
        return strategy;
    };

    StrategyCost cost = 0;

    for (auto* inputOp : _config.getVFOperations() | filtered([](mlir::Operation* op) {
                             return op->getNumOperands() > 1 && op->hasTrait<VPU::EltwiseOp>();
                         })) {
        for (auto operand : inputOp->getOperands()) {
            if (auto arg = operand.dyn_cast<mlir::BlockArgument>()) {
                auto previousOp = _config.getSubgraph().getOperand(arg.getArgNumber()).getDefiningOp();
                if (mlir::isa_and_nonnull<Const::DeclareOp>(previousOp) || !isCmxOperation(previousOp, false) ||
                    isPrevOperationEarlyScheduled(previousOp, _config.getSubgraph())) {
                    continue;
                }

                if (auto vfOp = mlir::dyn_cast<VPU::VerticalFusionOp>(previousOp)) {
                    previousOp = vfOp.getBody()->getTerminator()->getOperands().back().getDefiningOp();
                }

                auto operandType = operand.getType().cast<vpux::NDTypeInterface>();
                auto operandSize = operandType.getTotalAllocSize();
                auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(previousOp);
                if (clusteredOp != nullptr && clusteredOp->hasAttr(VPU::multiClusterStrategy)) {
                    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, operandType.getShape(),
                                                                  clusteredOp->getAttr(VPU::multiClusterStrategy)
                                                                          .cast<VPU::MultiClusterStrategyAttr>()
                                                                          .getValue());
                    auto clusterType = getDistributedOutputTypeFromOp(clusteredOp, operandType, numClusters);
                    operandSize = clusterType.cast<vpux::NDTypeInterface>().getTotalAllocSize();
                }

                if (!_vfScheduling->validate(_config, _vfTilingStorage, operandSize)) {
                    OutputTiling prevOpTiling;
                    if (previousOp->hasAttr(tilingStrategy)) {
                        auto prevOpStrategy =
                                parseIntArrayAttr<int64_t>(previousOp->getAttr(tilingStrategy).cast<mlir::ArrayAttr>());
                        auto tiles =
                                fillDividedTiles(previousOp, Shape(prevOpStrategy), getShape(previousOp->getResult(0)));
                        VPUX_THROW_WHEN(mlir::failed(tiles) || tiles.value().empty(),
                                        "Cannot get tiles {0} for the operation in VF {1}", prevOpStrategy, previousOp);
                        prevOpTiling = tiles.value();
                    }
                    const auto parentOpParams = VPUNNCostParameters(getStrategy(previousOp), prevOpTiling);
                    cost += costFunction->getSpillingWriteCost(previousOp, parentOpParams);
                }
            }
        }
    }

    _cachedCost = _cachedCost.value() + cost;
}

StrategyCost VFCase::getCost(const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log) {
    VPUX_THROW_WHEN(!isInitialized(), "Cannot get cost of uninitialized VF case");

    if (!_cachedCost.has_value()) {
        if (_vfTilingStorage == nullptr) {
            _vfTilingStorage = std::make_unique<TilingOperationStorage>();
            auto tilingDims = parseIntArrayAttr<int64_t>(getTiling());
            auto tilingStorage = calculateTilingRegions(_config.getSubgraph(), tilingDims, log, _vfTilingStorage);
            VPUX_THROW_WHEN(mlir::failed(tilingStorage), "Cannot get tiling regions for {0} and {1} tiles",
                            _config.getSubgraph(), tilingDims);
        }

        _cachedCost = _vfScheduling->getCost(_config, _tilingNumber, _vfTilingStorage, costFunction);
        addCMXWriteSpills(costFunction);
    }

    return _cachedCost.value();
}

bool VFCase::isInitialized() {
    return _vfScheduling != nullptr && (_tilingNumber > 1 || _config.isPotentiallyPipelined());
}

VFConfig& VFCase::getConfig() {
    return _config;
}

mlir::ArrayAttr VFCase::getTiling() const {
    auto outType = _config.getSubgraph()->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto tilingArray = SmallVector<int64_t>(outType.getRank(), 1);
    tilingArray[_axis.ind()] = _tilingNumber;

    return getIntArrayAttr(_config.getSubgraph().getContext(), tilingArray);
}

int64_t VFCase::getTilingNumber() const {
    return _tilingNumber;
}

void VFCase::approveScheduling() {
    VPUX_THROW_WHEN(!isInitialized(), "Cannot approve uninitialized VF case");

    _config.getSubgraph().setScenario(_vfScheduling->getType());
    _config.getSubgraph().setTilingStrategyAttr(getTiling());
}
