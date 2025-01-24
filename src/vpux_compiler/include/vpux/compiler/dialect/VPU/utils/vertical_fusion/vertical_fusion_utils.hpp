//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_pipeline_container.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_storage.hpp"

namespace vpux {
namespace VPU {

// min length of tensor by tiled axis. It limits number of tiles
// which we may increase in order to fit in CMX
constexpr int64_t MINIMUM_LENGTH_TILING = 4;

// The specific back-infer strategy type
enum class BackInferStrategy { TILING_DIM, TILING_STRATEGY };

// information about input and output tiles for operands and result
using VFOperationTiling = std::pair<InputTiling, TileInfo>;

// storage keeps connection between argument number of the block and biggest tile
// for parent operation of the block for each separate VF tile
using TilingStorage = VFContainer<size_t, TileInfo>;

// storage keeps connection between operation in the block and its information
// about input and output tiles for each VF tile
using TilingOperationStorage = VFContainer<mlir::Operation*, VFOperationTiling, llvm::less_second>;

// storage keeps information for each pipelined "container"
// in each "container" there are operations which might be executed in parallel
using VFContainerPipelineStorage = VFContainer<size_t, VFPipelineContainer>;

// function gets tiling information from VF subgraph and builds tiling info going up
// to arguments of the block
// it returns tiles for parent operations of the block and connection between them and block arguments
// for each VF tile
TilingStorage restoreTilingRegions(VPU::VerticalFusionOp vfOp, Logger log,
                                   const TilingOperationStorage::UPtr& opStorage);
// calculate tiling regions based on particular tiling strategy
mlir::FailureOr<TilingStorage> calculateTilingRegions(VPU::VerticalFusionOp vfOp, ArrayRef<int64_t> tilingStrategy,
                                                      Logger log, const TilingOperationStorage::UPtr& opStorage);
// calculate tiling regions based on known output tiles for last operation in the block
mlir::FailureOr<TilingStorage> calculateTilingRegions(VPU::VerticalFusionOp vfOp, const OutputTiling& tiles, Logger log,
                                                      const TilingOperationStorage::UPtr& opStorage);
// calculate recursively tiling regions for the block starting from last operation and known output tiles for it
// function builds connection between block arguments and tiles
// in case TilingOperationStorage pointer was passed, it filles in connection between each operation and
// its input and output tiles
mlir::FailureOr<TilingStorage> calculateTilingRegions(mlir::Operation* operation, const OutputTiling& tiles, Logger log,
                                                      const TilingOperationStorage::UPtr& opStorage,
                                                      std::optional<size_t> numTile = std::nullopt);

// calculate limit for number of tiles for set of operations
int64_t getTilingLimit(Dim axis, ArrayRef<mlir::Operation*> operations);

// get the maximal valid tiling strategy for VF block between the given range of tiling strategy
mlir::FailureOr<SmallVector<int64_t>> getMaximalValidTilingStrategyFromRange(
        VPU::VerticalFusionOp op, ArrayRef<int64_t> lowerTilingStrategy, ArrayRef<int64_t> upperTilingStrategy,
        Dim tilingAxis, TilingOperationStorage::UPtr& opStorage, Logger log);

// get the minimal valid tiling strategy for VF block between the given range of tiling strategy
mlir::FailureOr<SmallVector<int64_t>> getMinimalValidTilingStrategyFromRange(
        VPU::VerticalFusionOp op, ArrayRef<int64_t> lowerTilingStrategy, ArrayRef<int64_t> upperTilingStrategy,
        Dim tilingAxis, TilingOperationStorage::UPtr& opStorage, Logger log);

// get the tiling dimension according to the tiling strategy
// return nullopt if there is no tiling
std::optional<Dim> getVFTilingDim(ArrayRef<int64_t> tilingStrategy);

// get dim for tiling from strategy. in case there is no particular dimension,
// get it from available dimensions of all operations in the subgraph
mlir::FailureOr<Dim> getVFTilingDim(ArrayRef<int64_t> tilingStrategy, ArrayRef<mlir::Operation*> operations);

// get allowed dims for tiling
DimArr getAllowedDims(ArrayRef<mlir::Operation*> operations, Logger log);

// check if whole operation is in CMX
bool isCmxOperation(mlir::Operation* operation, const bool checkTilingType);

// check if previous operation will be early scheduled
bool isPrevOperationEarlyScheduled(mlir::Operation* prevOp, mlir::Operation* nextOp);

// get parent operation with pure view like operations bypassed
mlir::Operation* findParent(mlir::Value operand);

// check if tiling is spatial
bool isSpatialTiling(ArrayRef<int64_t> strategy);

// function merges operations to VF and returns the created subgraph
VPU::VerticalFusionOp fuseOpsInBlock(mlir::PatternRewriter& rewriter, VPU::VerticalFusionOp vfOp,
                                     mlir::Operation* prevOp, mlir::ArrayAttr tilingInfo = nullptr);

SmallVector<SmallVector<int64_t>> backInferVFTilingStrategy(
        VPU::VFConfig& config, ArrayRef<int64_t> tilingStrategy,
        std::unordered_map<mlir::Operation*, SmallVector<int64_t>>& opStrategyMap);

SmallVector<vpux::Dim> backInferVFTilingDim(VPU::VFConfig& config, vpux::Dim outputDim,
                                            std::unordered_map<mlir::Operation*, vpux::Dim>& opDimMap);

template <typename ArgType, typename ResultType>
ResultType backInfer(VPU::TilingViewLikeOpInterface opIf, ArgType tiling, VPU::BackInferStrategy strategy);

template <typename ArgType, typename ResultType>
SmallVector<ResultType> backInferVFTiling(VPU::VFConfig& config, ArgType outputTiling, BackInferStrategy strategy,
                                          std::unordered_map<mlir::Operation*, ResultType>& opTilingMap);

}  // namespace VPU
}  // namespace vpux
