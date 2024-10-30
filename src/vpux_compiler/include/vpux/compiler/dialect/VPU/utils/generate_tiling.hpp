//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

#include <mlir/IR/IRMapping.h>

namespace vpux {
namespace VPU {

constexpr float INPUT_OVERLAP_THRESHOLD = 2.0;

TilingMode getTilingSupportedMode(VPU::TilingBuilderOpInterface origOp, bool enablePrefetchTiling, Logger log);

mlir::FailureOr<OutputTiling> getLayerTilingStrategy(VPU::TilingBuilderOpInterface origOp, bool enablePrefetchTiling,
                                                     TilingMode& mode, Logger log);
mlir::FailureOr<OutputTiling> getLayerTilingStrategy(VPU::TilingBuilderOpInterface origOp, bool enablePrefetchTiling,
                                                     Logger log);

mlir::LogicalResult checkAndAlignActInputTiling(vpux::VPU::NCEOpInterface nceOp, InputTiling& inputTiling,
                                                vpux::Logger log);
mlir::Value reifyTile(VPU::TilingBuilderOpInterface origOp, const TileInfo& outputTile, mlir::OpBuilder& builder,
                      Logger log);
mlir::LogicalResult applyTileStrategy(VPU::TilingBuilderOpInterface origOp, const OutputTiling& tiles,
                                      mlir::PatternRewriter& rewriter, Logger log);
mlir::Operation* getParentComputeOp(mlir::Operation* op);
bool prefetchTilingConditionSatisfied(mlir::Operation* op, Logger log);
bool largeConstPipelineConditionSatisfied(mlir::Operation* op, Logger log);
bool hasMultiBranches(mlir::Operation* op);
Dim getHighestDimFromType(vpux::NDTypeInterface type);

bool archSupportsSwLayerTiling(VPU::ArchKind arch);
bool opNeedsTiling(mlir::Operation* op, bool enablePrefetchTiling, Logger log);
bool doesNCEOpChannelSatisfyWorkload(mlir::Operation* nceOp, const TileInfo& outputTile);
std::optional<DimArr> getSEPConvTilingOrder(mlir::Operation* op);
std::optional<std::pair<size_t, size_t>> getWorkLoadInformationForNCEWithSparseOutput(
        VPU::ArchKind arch, ArrayRef<Shape> perClusterShapes, ArrayRef<int64_t> supportedChannels);

/**
 * @brief Get the best tiling strategy based on the VPUNN cost model
 * @details If existing one-dimension tiling strategy, compare the costs and return the lowest-cost strategy.
 * Else, follow the tileDimOrder, increase the tile number in turn and return the first supported multi-dimension
 * tiling strategy
 *          One-dimension tiling strategy: only split on one dimension. e.g., [1, 2, 1, 1], [1, 1, 5, 1], etc.
 *          Multi-dimension tiling strategy: split on more than one dimension. e.g., [1, 2, 5, 1], [1, 1, 2, 5], etc.
 * @example An NCE Eltwise layer of shape [1, 2048, 14, 14]. It has supported one-dim tiling strategies on C, H and W
 *          the minimum tiling number is 2 for each dimension, i.e., the layer fits into CMX splitting by 2
 *          [1, 2, 1, 1], [1, 1, 2, 1], [1, 1, 1, 2]
 *          Corresponding VPUNN costs are 161551, 151587, and 159641
 *          tiling strategy [1, 1, 2, 1] is chosen as the best tiling because of the lowest VPUNN cost (151587)
 * @param op The target operation to tile
 * @param tilingMode Prefetching or pipelining or isolated
 * @param tileDimOrder The tiling order of dimensions when nested tiling is required (tiling on multiple dimensions)
 */
mlir::FailureOr<OutputTiling> getHWLayerTilingStrategyBasedOnCost(mlir::Operation* op, TilingMode tilingMode,
                                                                  DimArrRef tileDimOrder,
                                                                  const std::shared_ptr<LayerCostModel>& costModel,
                                                                  Logger log);

enum class EnableShaveDDRAccessOptimization { TRUE, FALSE, AUTO };

EnableShaveDDRAccessOptimization getShaveDDRAccessOptimizationMode(StringRef enableShaveDDRAccessOptimization);

}  // namespace VPU
}  // namespace vpux
