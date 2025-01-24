//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/cost_model.hpp"
#include "vpux/utils/core/dense_map.hpp"

#include <atomic>
#include <memory>

namespace vpux {
namespace VPU {

using OutputTilingCacheItem = mlir::FailureOr<OutputTiling>;

/*
Cache for possible tiling strategies for an operation with specified multi cluster strategy, tiling dim order and
tiling mode.
*/

class OpTilingCache {
public:
    ~OpTilingCache() = default;
    OpTilingCache(const OpTilingCache&) = delete;
    OpTilingCache& operator=(const OpTilingCache&) = delete;

    static OpTilingCache& instance() {
        static OpTilingCache cache;
        return cache;
    }

    void enableIfNecessary(bool enable);

    llvm::hash_code calculateOpHash(mlir::Operation* op, const std::optional<TilingMode>& mode = std::nullopt,
                                    const std::optional<DimArrRef>& dimOrder = std::nullopt,
                                    const std::optional<OutputTiling>& outputTile = std::nullopt);

    llvm::hash_code calculateVPUNNLayerHash(const VPUNN::DPULayer& vpunnLayer,
                                            const VPUNN::VPULayerStrategy& vpunnStrategy);

    mlir::FailureOr<OutputTiling> getHWLayerTilingStrategyWithTileDimOrder(mlir::Operation* op, TilingMode tilingMode,
                                                                           DimArrRef tileDimOrder, Logger log);

    std::optional<OutputTilingCacheItem> getOutputTiling(llvm::hash_code opHash, mlir::Operation* op,
                                                         ShapeRef outputShape);

    std::optional<SmallVector<uint32_t>> getOpDpuCost(llvm::hash_code opHash);

    std::optional<uint32_t> getVPUNNLayerCost(llvm::hash_code layerHash);

    void updateOutputTiling(const std::optional<llvm::hash_code>& opHash, mlir::Operation* op,
                            const mlir::FailureOr<OutputTiling>& outputTile);

    void updateOpDPUCost(llvm::hash_code opHash, ArrayRef<uint32_t> cost);

    void updateVPUNNLayerCost(llvm::hash_code layerHash, uint32_t cost);

    bool isCacheSupported(mlir::Operation* op);

    void cleanUp();

    void printStats(Logger& logger) const;

private:
    OpTilingCache() = default;

    std::optional<llvm::hash_code> calculateInputOutputModeHash(mlir::Operation* op,
                                                                const mlir::FailureOr<OutputTiling>& outputTiling);

    std::mutex _tilingMutex;
    std::mutex _dpuMutex;
    std::mutex _vpunnLayerMutex;
    DenseMap<llvm::hash_code, std::optional<Shape>> _tilingCache;
    DenseMap<llvm::hash_code, std::optional<llvm::hash_code>> _opHashToInputOutputModeHash;
    DenseMap<llvm::hash_code, SmallVector<uint32_t>> _opDpuCostCache;
    DenseMap<llvm::hash_code, uint32_t> _vpunnLayerCostCache;

    bool _enableCache{false};

    std::atomic<uint64_t> _tilingHitCount{0};
    std::atomic<uint64_t> _tilingAccessCount{0};

    std::atomic<uint64_t> _dpuCostHitCount{0};
    std::atomic<uint64_t> _dpuCostAccessCount{0};

    std::atomic<uint64_t> _vpunnLayerCostHitCount{0};
    std::atomic<uint64_t> _vpunnLayerCostAccessCount{0};
};
}  // namespace VPU
}  // namespace vpux
