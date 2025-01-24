//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/op_tiling_cache.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"

#include <mlir/IR/OperationSupport.h>

using namespace vpux;
using namespace VPU;

void OpTilingCache::enableIfNecessary(bool enable) {
    _enableCache = enable;
}

mlir::FailureOr<OutputTiling> OpTilingCache::getHWLayerTilingStrategyWithTileDimOrder(mlir::Operation* op,
                                                                                      TilingMode tilingMode,
                                                                                      DimArrRef tileDimOrder,
                                                                                      Logger log) {
    std::optional<llvm::hash_code> opHash;
    const auto useCache = isCacheSupported(op);
    if (useCache) {
        // NB: bounds information is used for tiling when dynamic shape is passed
        // TODO: E#113258 this logic can be put inside getShape impelementation
        const auto outputShape = [&op] {
            const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
            const auto origOutputShape = outputType.getShape();
            if (origOutputShape.isDynamic()) {
                const auto bounds =
                        parseIntArrayAttr<int64_t>(outputType.dyn_cast<vpux::BoundedTypeInterface>().getBounds());
                return vpux::Shape(bounds.begin(), bounds.end());
            }
            return vpux::Shape(origOutputShape.begin(), origOutputShape.end());
        }();

        opHash = calculateOpHash(op, tilingMode, tileDimOrder);
        auto cacheResult = getOutputTiling(opHash.value(), op, outputShape);
        if (cacheResult.has_value()) {
            return std::move(cacheResult.value());
        }
    }
    auto tilingStrategy = vpux::getHWLayerTilingStrategyWithTileDimOrder(op, tilingMode, tileDimOrder, log);
    if (useCache) {
        updateOutputTiling(opHash, op, tilingStrategy);
    }
    return tilingStrategy;
}

std::optional<OutputTilingCacheItem> OpTilingCache::getOutputTiling(llvm::hash_code opHash, mlir::Operation* op,
                                                                    ShapeRef outputShape) {
    if (!_enableCache) {
        return std::nullopt;
    }
    _tilingAccessCount++;
    std::lock_guard<std::mutex> lock(_tilingMutex);
    auto it = _tilingCache.find(opHash);
    if (it == _tilingCache.end()) {
        return std::nullopt;
    }

    auto cachedInputOutputModeHash = _opHashToInputOutputModeHash.find(opHash);
    if (cachedInputOutputModeHash == _opHashToInputOutputModeHash.end()) {
        return std::nullopt;
    }
    auto nTilesOnDim = it->second;
    mlir::FailureOr<OutputTiling> tilingStrategy = mlir::failure();
    if (nTilesOnDim.has_value()) {
        tilingStrategy = fillDividedTiles(op, nTilesOnDim.value(), outputShape);
    }
    auto modeHash = calculateInputOutputModeHash(op, tilingStrategy);
    if (modeHash != cachedInputOutputModeHash->second) {
        // Disitrubted output mode is changed, cache is invalid
        return std::nullopt;
    }

    _tilingHitCount++;
    return tilingStrategy;
}

std::optional<SmallVector<uint32_t>> OpTilingCache::getOpDpuCost(llvm::hash_code opHash) {
    if (!_enableCache) {
        return std::nullopt;
    }
    _dpuCostAccessCount++;
    std::lock_guard<std::mutex> lock(_dpuMutex);
    auto it = _opDpuCostCache.find(opHash);
    if (it == _opDpuCostCache.end()) {
        return std::nullopt;
    }
    _dpuCostHitCount++;
    return it->second;
}

std::optional<uint32_t> OpTilingCache::getVPUNNLayerCost(llvm::hash_code layerHash) {
    if (!_enableCache) {
        return std::nullopt;
    }
    _vpunnLayerCostAccessCount++;
    std::lock_guard<std::mutex> lock(_vpunnLayerMutex);
    auto it = _vpunnLayerCostCache.find(layerHash);
    if (it == _vpunnLayerCostCache.end()) {
        return std::nullopt;
    }
    _vpunnLayerCostHitCount++;
    return it->second;
}

void OpTilingCache::printStats(Logger& logger) const {
    if (!_enableCache) {
        return;
    }

    logger.info("Tiling cache hit : {0}", _tilingHitCount);
    logger.info("Tiling cache miss : {0}", _tilingAccessCount - _tilingHitCount);
    if (_tilingAccessCount != 0) {
        logger.info("Tiling cache hit rate: {0}%", _tilingHitCount * 100.0 / _tilingAccessCount);
    }

    logger.info("DPU cost cache hit : {0}", _dpuCostHitCount);
    logger.info("DPU cost cache miss : {0}", _dpuCostAccessCount - _dpuCostHitCount);
    if (_dpuCostAccessCount != 0) {
        logger.info("DPU cost cache hit rate: {0}%", _dpuCostHitCount * 100.0 / _dpuCostAccessCount);
    }

    logger.info("VPUNNLayer cost cache hit : {0}", _vpunnLayerCostHitCount);
    logger.info("VPUNNLayer cost cache miss : {0}", _vpunnLayerCostAccessCount - _vpunnLayerCostHitCount);
    if (_vpunnLayerCostAccessCount != 0) {
        logger.info("VPUNNLayer cost cache hit rate: {0}%",
                    _vpunnLayerCostHitCount * 100.0 / _vpunnLayerCostAccessCount);
    }
}

void OpTilingCache::updateOutputTiling(const std::optional<llvm::hash_code>& opHash, mlir::Operation* op,
                                       const mlir::FailureOr<OutputTiling>& outputTiling) {
    if (!_enableCache || !opHash.has_value()) {
        return;
    }

    auto outputModeHash = calculateInputOutputModeHash(op, outputTiling);
    std::lock_guard<std::mutex> lock(_tilingMutex);
    if (mlir::failed(outputTiling)) {
        _tilingCache.insert({opHash.value(), std::nullopt});
    } else {
        VPUX_THROW_WHEN(outputTiling.value().empty(), "Output tiling is empty for op {0}", op->getLoc());
        _tilingCache.insert({opHash.value(), outputTiling.value().front().axis});
    }
    _opHashToInputOutputModeHash.insert({opHash.value(), outputModeHash});
}

void OpTilingCache::updateOpDPUCost(llvm::hash_code opHash, ArrayRef<uint32_t> dpuCosts) {
    if (!_enableCache) {
        return;
    }
    std::lock_guard<std::mutex> lock(_dpuMutex);
    _opDpuCostCache[opHash] = SmallVector<uint32_t>{dpuCosts.begin(), dpuCosts.end()};
}

void OpTilingCache::updateVPUNNLayerCost(llvm::hash_code layerHash, uint32_t cost) {
    if (!_enableCache) {
        return;
    }
    std::lock_guard<std::mutex> lock(_vpunnLayerMutex);
    _vpunnLayerCostCache[layerHash] = cost;
}

void OpTilingCache::cleanUp() {
    _tilingAccessCount = 0;
    _tilingHitCount = 0;
    _dpuCostAccessCount = 0;
    _dpuCostHitCount = 0;
    _vpunnLayerCostAccessCount = 0;
    _vpunnLayerCostHitCount = 0;

    {
        std::lock_guard<std::mutex> lock(_tilingMutex);
        _tilingCache.clear();
        _opHashToInputOutputModeHash.clear();
    }
    {
        std::lock_guard<std::mutex> lock(_dpuMutex);
        _opDpuCostCache.clear();
    }

    std::lock_guard<std::mutex> lock(_vpunnLayerMutex);
    _vpunnLayerCostCache.clear();
}

bool OpTilingCache::isCacheSupported(mlir::Operation* op) {
    if (!_enableCache) {
        return false;
    }
    // For sparse op, it's difficult to calculate the hash for the operand's type, so disable it for now. See E#150101
    // for details.
    return !llvm::any_of(op->getOperands(), [](mlir::Value operand) {
        return mlir::isa_and_nonnull<VPU::GroupSparseTensorOp>(operand.getDefiningOp());
    });
}

llvm::hash_code OpTilingCache::calculateOpHash(mlir::Operation* op, const std::optional<TilingMode>& mode,
                                               const std::optional<DimArrRef>& dimOrder,
                                               const std::optional<OutputTiling>& outputTiling) {
    // The hash result is composed of the hash of the op, the dim order, the output tiling and the tiling mode.
    // For the op's hash, it will be calculated based on the op's type, input/output type and its attributes.
    // If the tiling mode is PREFETCHING, the hash will also include the hash of the parent compute op since the parent
    // op will affect the decision of the tiling result.
    auto hashValueType = [](mlir::Value operand) {
        auto type = mlir::cast<vpux::NDTypeInterface>(operand.getType());
        return llvm::hash_value(llvm::formatv("{0} {1} {2} {3}", type.getShape(), type.getStrides(),
                                              type.getElemTypeSize(), type.getTotalAllocSize())
                                        .str());
    };

    auto opHash = mlir::OperationEquivalence::computeHash(op,
                                                          /*hashOperands=*/hashValueType,
                                                          /*hashResults=*/mlir::OperationEquivalence::ignoreHashValue,
                                                          mlir::OperationEquivalence::IgnoreLocations);

    if (dimOrder.has_value()) {
        auto getDimArrHash = [](DimArrRef dimOrder) {
            return llvm::hash_value(llvm::formatv("{0}", dimOrder).str());
        };
        opHash = llvm::hash_combine(opHash, getDimArrHash(dimOrder.value()));
    }

    if (outputTiling.has_value()) {
        opHash = llvm::hash_combine(opHash, llvm::hash_value(llvm::formatv("{0}", outputTiling.value()).str()));
    }

    if (mode.has_value()) {
        opHash = llvm::hash_combine(opHash, mode.value());
        if (mode.value() == TilingMode::PREFETCHING) {
            if (auto parentOp = VPU::getParentComputeOp(op)) {
                opHash = llvm::hash_combine(opHash, mlir::OperationEquivalence::computeHash(
                                                            parentOp, mlir::OperationEquivalence::ignoreHashValue,
                                                            mlir::OperationEquivalence::ignoreHashValue,
                                                            mlir::OperationEquivalence::IgnoreLocations));
            }
        }
    }
    return opHash;
}

llvm::hash_code OpTilingCache::calculateVPUNNLayerHash(const VPUNN::DPULayer& vpunnLayer,
                                                       const VPUNN::VPULayerStrategy& vpunnStrategy) {
    std::ostringstream layerStream;
    layerStream << vpunnLayer;
    layerStream << vpunnStrategy;
    return llvm::hash_value(layerStream.str());
}

std::optional<llvm::hash_code> OpTilingCache::calculateInputOutputModeHash(
        mlir::Operation* op, const mlir::FailureOr<OutputTiling>& outputTiling) {
    if (!op->hasAttr(vpux::multiClusterStrategy)) {
        return std::nullopt;
    }
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op);
    VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} is not a clustered op but has MultiClusterStrategy attr",
                    op->getLoc());
    auto mcStrategy = clusteredOp.getMultiClusterStrategy().value();
    auto outputType = mlir::dyn_cast<vpux::NDTypeInterface>(op->getResult(0).getType());

    SmallVector<VPU::DistributionMode> inputOutputMode;
    inputOutputMode.push_back(getActivationTensorDistributionMode(clusteredOp, mcStrategy));
    if (mlir::succeeded(outputTiling)) {
        auto uniqueTiles = VPU::getUniqueShapeTilingCandidates(op, outputTiling.value(), Logger::global());
        for (auto& outputTile : uniqueTiles) {
            const auto outputTileType = outputType.extractDenseTile(outputTile.offsets, outputTile.shape);
            inputOutputMode.push_back(getOutputTensorDistributionMode(clusteredOp, mcStrategy, outputTileType));
        }
    } else {
        inputOutputMode.push_back(getOutputTensorDistributionMode(clusteredOp, mcStrategy, outputType));
    }
    return llvm::hash_combine_range(inputOutputMode.begin(), inputOutputMode.end());
}
