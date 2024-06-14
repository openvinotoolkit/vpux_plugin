//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/vf_axis_increment.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion_utils.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/SetOperations.h>
#include <llvm/ADT/SmallSet.h>

#include <mlir/IR/IRMapping.h>

using namespace vpux;
using namespace VPU;

namespace {

//
// MergeVFRegionRewriter
//

class MergeVFRegionRewriter final : public mlir::OpRewritePattern<VPU::VerticalFusionOp> {
public:
    MergeVFRegionRewriter(mlir::MLIRContext* ctx, bool enableVerticalFusionPipelining, bool enablePrefetchTiling,
                          const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log)
            : mlir::OpRewritePattern<VPU::VerticalFusionOp>(ctx),
              _enableVerticalFusionPipelining(enableVerticalFusionPipelining),
              _enablePrefetchTiling(enablePrefetchTiling),
              _vpunnCostFunction(costFunction),
              _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(VPU::VerticalFusionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool checkVFCostFunction(VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currentOp,
                             VPU::VerticalFusionOp mergedVFOp, mlir::ArrayAttr tiling) const;
    int64_t getMaxDimLimit(const Dim axis, ArrayRef<mlir::Operation*> operation) const;
    bool getOptimalTilingStrategy(SmallVector<int64_t>& tilingArray, const Dim dim, const int64_t minTiles,
                                  const int64_t maxTiles, VFConfig& config) const;
    bool waitOtherUsers(VPU::VerticalFusionOp newBlock, VPU::VerticalFusionOp parentVFOp) const;
    mlir::ArrayAttr getVFTilingInfo(VPU::VerticalFusionOp newBlock, VPU::VerticalFusionOp parentVFOp,
                                    VPU::VerticalFusionOp mergedVFOp) const;
    bool alignMCTiling(VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp) const;
    mlir::FailureOr<Dim> getTilingAxis(SmallVector<int64_t>& tilingStrategy, VPU::VerticalFusionOp prevOp,
                                       VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp mergedVFOp) const;

    void fuseBlocks(mlir::PatternRewriter& rewriter, VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp mergedOp,
                    mlir::ArrayAttr tilingInfo) const;

    bool _enableVerticalFusionPipelining = false;
    bool _enablePrefetchTiling = true;
    const std::unique_ptr<VPU::LayerVPUNNCost>& _vpunnCostFunction;
    Logger _log;
};

inline bool hasTiling(const ArrayRef<int64_t> tilingInfo) {
    return llvm::any_of(tilingInfo, [](auto i) {
        return i != 1;
    });
}

bool MergeVFRegionRewriter::alignMCTiling(VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp) const {
    const auto prevBlock = prevOp.getBody();
    const auto parentVFOp = currentOp.getBody();

    auto newOps = prevBlock->getOps<VPU::VerticalFusionOpInterface>();
    auto oldOps = parentVFOp->getOps<VPU::VerticalFusionOpInterface>();

    if (newOps.empty() || oldOps.empty()) {
        return false;
    }
    // all ops have same multicluster strategies or don't have them at all
    // so, compare only first operations in each block
    const auto isClusteredOp = [](auto op) {
        return mlir::dyn_cast<VPU::ClusteredOpInterface>(op.getOperation()) != nullptr;
    };
    const auto firstOldClusterOp = llvm::find_if(oldOps, isClusteredOp);
    const auto firstNewClusterOp = llvm::find_if(newOps, isClusteredOp);

    if (firstOldClusterOp != oldOps.end() && firstNewClusterOp != newOps.end()) {
        const auto oldBlockStrategy =
                mlir::dyn_cast<VPU::ClusteredOpInterface>(**firstOldClusterOp).getMultiClusterStrategy();
        const auto newBlockStrategy =
                mlir::dyn_cast<VPU::ClusteredOpInterface>(**firstNewClusterOp).getMultiClusterStrategy();

        // if only one strategy is defined - blocks don't match
        // in case both strategies are defined, they must be same
        if (oldBlockStrategy.has_value() ^ newBlockStrategy.has_value()) {
            return false;
        }

        if (oldBlockStrategy.has_value() && newBlockStrategy.has_value() &&
            oldBlockStrategy.value() != newBlockStrategy.value()) {
            return false;
        }
    }

    return true;
}

Byte getRequiredWeightsMemory(ArrayRef<VPU::VerticalFusionOpInterface> ops) {
    auto weightsMem = Byte(0);
    for (auto& op : ops) {
        if (mlir::isa<VPU::NCEOpInterface>(*op)) {
            auto outputShape = op->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape();
            weightsMem += getRequiredCMXForWeight(op, TileInfo(outputShape));
        }
    }

    return weightsMem;
}

/*
 Function checks if two blocks suit to be merged in one on following criterias:
 1. Number of operations doesn't exceed the limit
 2. In case there is only one operation in the block, it might be merged as first op in the block
 3. All multicluster strategies are same for both blocks if there are any
 4. Required CMX memory by constant weights shouldn't exceed the size of the whole memory
*/
bool MergeVFRegionRewriter::checkVFCostFunction(VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currentOp,
                                                VPU::VerticalFusionOp mergedVFOp, mlir::ArrayAttr tiling) const {
    VPUX_THROW_WHEN(tiling == nullptr, "Incorrect tiling strategy for VF");

    const auto prevBlock = prevOp.getBody();
    const auto currentVFOp = currentOp.getBody();

    auto newOps = currentVFOp->getOps<VPU::VerticalFusionOpInterface>();
    auto oldOps = prevBlock->getOps<VPU::VerticalFusionOpInterface>();

    if (newOps.empty() || oldOps.empty()) {
        return false;
    }

    // compare the cost between merged VF Subgraph and 2 subgraphs with the spill
    const auto prevCost = getVFCost(_vpunnCostFunction, prevOp, _log, _enablePrefetchTiling);
    const auto currentCost = getVFCost(_vpunnCostFunction, currentOp, _log, _enablePrefetchTiling);

    // simply decide if there is tiling for parents
    const auto prevTilingStrategy = parseIntArrayAttr<int64_t>(prevOp.getTilingStrategy());
    const auto currentTilingStrategy = parseIntArrayAttr<int64_t>(currentOp.getTilingStrategy());
    StrategyCost spillingCost = 0;
    const auto moreThanOne = [](auto value) {
        return value > 1;
    };

    const auto isSpatialTiling = [](auto& strategy) {
        if (strategy.size() <= Dims4D::Act::numSpatialDims) {
            return false;
        }

        for (auto index : irange(Dims4D::Act::numSpatialDims)) {
            if (strategy[Dims4D::Act::getSpatialDim(index).ind()] > 1) {
                return true;
            }
        }

        return false;
    };
    const auto spillBetween =
            (llvm::any_of(prevTilingStrategy, moreThanOne) && isSpatialTiling(prevTilingStrategy)) ||
            (llvm::any_of(currentTilingStrategy, moreThanOne) && isSpatialTiling(currentTilingStrategy));

    const auto getStrategy = [](auto* operation) -> VPU::MultiClusterStrategy {
        auto strategy = VPU::MultiClusterStrategy::Clustering;
        if (auto mcOperation = mlir::dyn_cast<VPU::ClusteredOpInterface>(operation)) {
            strategy = mcOperation.getMultiClusterStrategy().value_or(strategy);
        }
        return strategy;
    };

    if (spillBetween) {
        // for each operation from previous subgraph which has users in current subgraph
        // calculate spill cost
        const auto lastOp = prevBlock->getTerminator()->getOperands().back().getDefiningOp();
        auto mcStrategy = getStrategy(lastOp);

        const auto prevTiles = fillDividedTiles(prevOp, Shape(prevTilingStrategy), getShape(lastOp->getResult(0)));

        const auto lastOpParams = VPUNNCostParameters(mcStrategy, prevTiles.value());
        spillingCost += _vpunnCostFunction->getSpillingWriteCost(lastOp, lastOpParams);

        for (auto arg : currentVFOp->getArguments()) {
            auto operand = currentOp.getOperand(arg.getArgNumber());
            auto findOperand = [&](mlir::Value value) -> bool {
                return value == arg;
            };
            if (operand.getDefiningOp() == prevOp) {
                for (auto* user : arg.getUsers()) {
                    mcStrategy = getStrategy(user);
                    const auto currentTiles =
                            fillDividedTiles(user, Shape(currentTilingStrategy), getShape(user->getResult(0)));

                    const auto userParams = VPUNNCostParameters(mcStrategy, currentTiles.value());
                    spillingCost += _vpunnCostFunction->getSpillingReadCost(user, userParams, prevOp.getOperation(),
                                                                            findOperand);
                }
            }
        }
    }

    // create new VF, in case the cost is worse, delete it
    // E-121586
    currentOp.getResult(0).replaceAllUsesWith(mergedVFOp.getResult(0));
    auto mergedVFCost = getVFCost(_vpunnCostFunction, mergedVFOp, _log, _enablePrefetchTiling, tiling);
    mergedVFOp.getResult(0).replaceAllUsesWith(currentOp.getResult(0));

    const auto spillAfter = llvm::any_of(currentTilingStrategy, moreThanOne);
    const auto isCmxOperation = [](auto* operation) {
        return mlir::isa_and_nonnull<VPU::TilingInfoOpInterface>(operation) && !operation->hasAttr(tilingStrategy);
    };

    if (!spillAfter) {
        if (!currentVFOp->getUsers().empty() && llvm::all_of(currentVFOp->getUsers(), isCmxOperation)) {
            // add spill after the last op to DDR
            const auto lastOp = currentVFOp->getTerminator()->getOperands().back().getDefiningOp();
            VPUX_THROW_WHEN(lastOp == nullptr, "Cannot get the last operation in VF {0}", currentOp);
            auto tiles = fillDividedTiles(currentOp, Shape(currentTilingStrategy), getShape(currentOp->getResult(0)));
            VPUX_THROW_WHEN(mlir::failed(tiles) || tiles.value().empty(),
                            "Cannot get tiles {0} for the operation in VF {1}", currentTilingStrategy, currentOp);

            auto types = getTileTypes(lastOp, tiles.value().back());
            VPUX_THROW_WHEN(types.empty(), "Cannot get type for operation {0} tile {1}", *lastOp, tiles.value().back());

            mergedVFCost += types.back().getTotalAllocSize().count();
        }
    }

    if (llvm::none_of(prevTilingStrategy, moreThanOne)) {
        auto currentConfig = VFConfig(currentOp);
        auto currentOperations = currentConfig.getVFOperations();
        const auto isBlockArg = [](mlir::Value value) -> bool {
            return value.isa<mlir::BlockArgument>();
        };
        DenseMap<mlir::Value, Byte> operandsSpills;
        for (auto* inputOp : currentOperations | filtered([](mlir::Operation* op) {
                                 return op->hasTrait<VPU::EltwiseOp>();
                             })) {
            // for input operations which have one of operands not in
            // VFblock arguments
            // calculate spilling
            if (!llvm::any_of(inputOp->getOperands(), isBlockArg)) {
                continue;
            }
            for (auto operand : inputOp->getOperands()) {
                if (auto arg = operand.dyn_cast<mlir::BlockArgument>()) {
                    auto previousOp = currentOp.getOperand(arg.getArgNumber()).getDefiningOp();
                    if (mlir::isa_and_nonnull<Const::DeclareOp, VPU::VerticalFusionOp>(previousOp) ||
                        previousOp == prevOp || !isCmxOperation(previousOp)) {
                        continue;
                    }

                    auto operandCost = operand.getType().cast<vpux::NDTypeInterface>().getTotalAllocSize();
                    auto curOpStorage = std::make_unique<TilingOperationStorage>();
                    VPU::restoreTilingRegions(currentOp, _log, curOpStorage);
                    if (!validateCMXSize(currentConfig, curOpStorage, _log, operandCost)) {
                        mergedVFCost += 2 * operandCost.count();
                    }
                }
            }
        }
    }

    if (mergedVFCost > prevCost + currentCost + spillingCost) {
        return false;
    }

    // the memory required by constant weights should be less than the threshold
    // otherwise there might be spilling for the weights
    auto weightsMem = getRequiredWeightsMemory(to_small_vector(oldOps));
    weightsMem += getRequiredWeightsMemory(to_small_vector(newOps));
    const auto totalCMXSize = VPU::getTotalCMXSize(prevOp.getOperation()).count() * VF_WEIGHTS_RATIO;
    if (totalCMXSize <= weightsMem.count()) {
        _log.trace("Required weights memory exceeds the total memory size");
        return false;
    }

    return true;
}

/*
 As soon as we don't have logic right now for excluding operations or break subgraph
 check in advance that all users or previous block will be merged to current one
*/
bool MergeVFRegionRewriter::waitOtherUsers(VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currentOp) const {
    if (prevOp->hasOneUse()) {
        return true;
    }

    for (auto user : prevOp->getUsers()) {
        if (!mlir::isa<VPU::VerticalFusionOp>(user)) {
            return false;
        }
        if (user == currentOp) {
            continue;
        }

        const auto userGoToRegion = llvm::any_of(user->getUsers(), [&](auto current) {
            return current != currentOp;
        });

        if (userGoToRegion) {
            return false;
        }
    }

    return true;
}

bool MergeVFRegionRewriter::getOptimalTilingStrategy(SmallVector<int64_t>& tilingArray, const Dim dim,
                                                     const int64_t minTiles, const int64_t maxTiles,
                                                     VFConfig& config) const {
    if (minTiles > maxTiles || maxTiles == 1) {
        return false;
    }

    auto minNTiles = minTiles;
    auto maxNTiles = maxTiles;

    auto outType = config.getSubgraph()->getResult(0).getType().cast<vpux::NDTypeInterface>();
    tilingArray = SmallVector<int64_t>(outType.getRank(), 1);
    tilingArray[dim.ind()] = minNTiles;

    if (minTiles == maxTiles) {
        auto curOpStorage = std::make_unique<TilingOperationStorage>();
        auto tilingRegions = VPU::calculateTilingRegions(config.getSubgraph(), tilingArray, _log, curOpStorage);

        return mlir::succeeded(tilingRegions) && validateCMXSize(config, curOpStorage, _log);
    }

    auto tilingMaxStrategy = SmallVector<int64_t>(outType.getRank(), 1);
    tilingMaxStrategy[dim.ind()] = maxNTiles;

    auto opStorage = std::make_unique<TilingOperationStorage>();

    auto getValidStrategy = VPU::getMinimalValidTilingStrategyFromRange(config.getSubgraph(), tilingArray,
                                                                        tilingMaxStrategy, dim, opStorage, _log);

    if (mlir::failed(getValidStrategy)) {
        return false;
    }

    tilingArray = getValidStrategy.value();
    minNTiles = tilingArray[dim.ind()];

    if (validateCMXSize(config, opStorage, _log)) {
        return true;
    }

    getValidStrategy = VPU::getMaximalValidTilingStrategyFromRange(config.getSubgraph(), tilingArray, tilingMaxStrategy,
                                                                   dim, opStorage, _log);

    if (mlir::failed(getValidStrategy) || !validateCMXSize(config, opStorage, _log)) {
        return false;
    }

    tilingMaxStrategy = getValidStrategy.value();
    maxNTiles = tilingMaxStrategy[dim.ind()];

    auto axisIncrement = getVFAxisIncrement(dim);
    VPUX_THROW_WHEN(axisIncrement == nullptr, "Cannot get functions to get values for axis {0}", dim);

    auto nextValueFromMin = minNTiles;
    axisIncrement->increasedValue(nextValueFromMin, maxNTiles);

    while (minNTiles < maxNTiles) {
        auto currentNTiles = axisIncrement->getMiddleValue(minNTiles, maxNTiles);

        if (maxNTiles == nextValueFromMin) {
            break;
        }

        if (currentNTiles == minNTiles) {
            return false;
        }

        tilingMaxStrategy[dim.ind()] = maxNTiles;
        tilingArray[dim.ind()] = currentNTiles;

        auto getValidTilingStrategy = VPU::getMinimalValidTilingStrategyFromRange(
                config.getSubgraph(), tilingArray, tilingMaxStrategy, dim, opStorage, _log);
        if (mlir::failed(getValidTilingStrategy)) {
            return false;
        }

        tilingArray = getValidTilingStrategy.value();
        currentNTiles = tilingArray[dim.ind()];

        if (currentNTiles == maxNTiles) {
            break;
        }

        if (validateCMXSize(config, opStorage, _log)) {
            maxNTiles = currentNTiles;
        } else {
            minNTiles = currentNTiles;
        }

        nextValueFromMin = minNTiles;
        axisIncrement->increasedValue(nextValueFromMin, maxNTiles);
    }

    return true;
}

mlir::FailureOr<Dim> MergeVFRegionRewriter::getTilingAxis(SmallVector<int64_t>& tilingArray,
                                                          VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currentOp,
                                                          VPU::VerticalFusionOp mergedVFOp) const {
    const auto currentTiling = parseIntArrayAttr<int64_t>(currentOp.getTilingStrategy());
    const auto prevTiling = parseIntArrayAttr<int64_t>(prevOp.getTilingStrategy());

    VPUX_THROW_WHEN(currentTiling.size() != prevTiling.size(),
                    "Tiling info rank of current block {0} is not equal to tiling info rank of previous block {1}",
                    currentTiling.size(), prevTiling.size());
    auto currentConfig = VFConfig(currentOp, _enableVerticalFusionPipelining);
    auto prevConfig = VFConfig(prevOp, _enableVerticalFusionPipelining);

    auto curAxis = getVFTilingDim(currentTiling, currentConfig.getVFOperations());
    auto prevAxis = getVFTilingDim(prevTiling, prevConfig.getVFOperations());

    if (mlir::failed(curAxis) || mlir::failed(prevAxis)) {
        return mlir::failure();
    }

    bool curHasTiling = hasTiling(currentTiling);
    bool prevHasTiling = hasTiling(prevTiling);
    // in case both subgraphs have tiling, check if they match
    // if there is only one subgraph with tiling, check if it's allowed
    // to tile second one with such axis
    // if both doesn't have tiling, check if there is at least one
    // allowed axis for both of them

    auto vfConfig = VFConfig(mergedVFOp, _enableVerticalFusionPipelining);

    if (!prevHasTiling && !curHasTiling && !vfConfig.isPipelined()) {
        const auto filterNCENotEltwiseLike = [](mlir::Operation* op) {
            return mlir::isa<VPU::NCEOpInterface>(op) && !op->hasTrait<VPU::EltwiseOp>();
        };
        const auto filterSWKernels = [](mlir::Operation* op) {
            return mlir::isa<VPU::SWOpInterface>(op);
        };
        const auto nonViewLike = [](mlir::Operation* op) {
            return !mlir::isa_and_nonnull<VPU::TilingViewLikeOpInterface>(op);
        };
        // when pipeline case is generic this check is enough to prevent VF
        // but now we check additionally that there are no operations
        // with different executors
        auto checkedOperations = vfConfig.getVFOperations() | filtered(nonViewLike);
        if ((llvm::all_of(checkedOperations, filterNCENotEltwiseLike) ||
             llvm::all_of(checkedOperations, filterSWKernels))) {
            return mlir::failure();
        }
    }

    auto getVFTilingStrategy = [this](SmallVector<int64_t>& tilingArray, const SmallVector<int64_t>& currentTiling,
                                      const SmallVector<int64_t>& prevTiling, const Dim dim, VFConfig& config) -> bool {
        const auto maxTiles = getTilingLimit(dim, config.getVFOperations());
        const auto minTiles = std::max(currentTiling[dim.ind()], prevTiling[dim.ind()]);

        // get optimal tiling strategy for the 1st time
        auto result = getOptimalTilingStrategy(tilingArray, dim, minTiles, maxTiles, config);
        if (result) {
            return tilingArray[dim.ind()] != 1 || config.isPotentiallyPipelined();
        }

        // if it failed to get optimal stratey for the first time with VF pipelining
        // try to get strategy once again without VF pipelining
        // it might success becasue VF requires smaller CMX memory size without VF pipelining
        if (config.isPipelined()) {
            config.disableVFPipeline();
            result = getOptimalTilingStrategy(tilingArray, dim, minTiles, maxTiles, config);
            config.restoreVFPipeline();

            // if we don't need tiling, VF is inefficient without pipelining
            return result && tilingArray[dim.ind()] != 1;
        }

        return false;
    };

    if (curHasTiling && prevHasTiling && curAxis.value() == prevAxis.value()) {
        if (!getVFTilingStrategy(tilingArray, currentTiling, prevTiling, curAxis.value(), vfConfig)) {
            return mlir::failure();
        }

        return curAxis.value();
    }

    DimArr allowedDims = getAllowedDims(vfConfig.getVFOperations(), _log);
    if (allowedDims.empty()) {
        return mlir::failure();
    }

    StrategyCost bestCost = std::numeric_limits<StrategyCost>::max();
    mlir::FailureOr<Dim> axis;
    for (auto dim : allowedDims) {
        // calculate optimal number of tiles for that dim
        SmallVector<int64_t> tilingAxisArray;
        if (!getVFTilingStrategy(tilingAxisArray, currentTiling, prevTiling, dim, vfConfig)) {
            continue;
        }

        // get vpunncost
        StrategyCost cost = getVFCost(_vpunnCostFunction, mergedVFOp, _log, _enablePrefetchTiling,
                                      getIntArrayAttr(getContext(), tilingAxisArray));
        // compare cost, choose best strategy

        if (cost < bestCost) {
            bestCost = cost;
            axis = dim;
            tilingArray = std::move(tilingAxisArray);
        }
    }
    return axis;
}

/*
 There are several steps in order to adjust tiling in order to get it fit in CMX
 1. Restore tiles for operation from current block
 2. Match block arguments and tiles
 3. Restore tiles for previous block starting from operations which are operands of current block
 4. In case some operations doesn't fit in CMX, try to increase number of tiles by the limit
 5. CMX memory used percentage by the largest operation shouldn't exceed VF_LARGEST_OP_MEM_RATIO to prevent spilling
*/
mlir::ArrayAttr MergeVFRegionRewriter::getVFTilingInfo(VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currentOp,
                                                       VPU::VerticalFusionOp mergedVFOp) const {
    if (!alignMCTiling(currentOp, prevOp)) {
        return nullptr;
    }

    SmallVector<int64_t> tilingArray;
    currentOp.getResult(0).replaceAllUsesWith(mergedVFOp.getResult(0));
    const auto axis = getTilingAxis(tilingArray, prevOp, currentOp, mergedVFOp);
    mergedVFOp.getResult(0).replaceAllUsesWith(currentOp.getResult(0));

    if (mlir::failed(axis)) {
        return nullptr;
    }

    for (auto operation : currentOp.getBody()->getOps<VPU::VerticalFusionOpInterface>()) {
        auto restrictedAxes = operation.restrictedFusionAxes();
        if (llvm::find(restrictedAxes, axis.value()) != restrictedAxes.end()) {
            return nullptr;
        }
    }

    return getIntArrayAttr(currentOp.getContext(), tilingArray);
}

void MergeVFRegionRewriter::fuseBlocks(mlir::PatternRewriter& rewriter, VPU::VerticalFusionOp currentOp,
                                       VPU::VerticalFusionOp mergedOp, mlir::ArrayAttr tilingInfo) const {
    mergedOp.setTilingStrategyAttr(tilingInfo);
    rewriter.replaceOp(currentOp, mergedOp.getResult(0));
}

mlir::LogicalResult MergeVFRegionRewriter::matchAndRewrite(VPU::VerticalFusionOp vfOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Vertical fusion region {0}", vfOp);

    VPU::VerticalFusionOp vfBlock = nullptr;
    VPU::VerticalFusionOp parentVFOp = nullptr;
    mlir::ArrayAttr tilingInfo = nullptr;
    for (auto operand : vfOp->getOperands()) {
        parentVFOp = operand.getDefiningOp<VPU::VerticalFusionOp>();
        vfBlock = nullptr;

        if (parentVFOp == nullptr) {
            continue;
        }

        _log.trace("Analyze vf region {0}", parentVFOp);

        const bool allInOldBlock = llvm::all_of(parentVFOp->getUsers(), [&](auto user) {
            return user == vfOp;
        });
        if (!allInOldBlock) {
            if (waitOtherUsers(parentVFOp, vfOp)) {
                continue;
            }
            return mlir::failure();
        }

        vfBlock = fuseOpsInBlock(rewriter, vfOp, parentVFOp.getOperation());
        tilingInfo = getVFTilingInfo(parentVFOp, vfOp, vfBlock);
        if (tilingInfo == nullptr) {
            rewriter.eraseOp(vfBlock);
            return mlir::failure();
        }

        if (!checkVFCostFunction(parentVFOp, vfOp, vfBlock, tilingInfo)) {
            rewriter.eraseOp(vfBlock);
            return mlir::failure();
        }

        break;
    }

    if (vfBlock == nullptr) {
        return mlir::failure();
    }

    _log.trace("Merged subgraph {0}", vfBlock);
    fuseBlocks(rewriter, vfOp, vfBlock, tilingInfo);

    return mlir::success();
}

//
// MergeVfSubgraphsPass
//

class MergeVfSubgraphsPass final : public MergeVfSubgraphsBase<MergeVfSubgraphsPass> {
public:
    explicit MergeVfSubgraphsPass(bool enableVerticalFusionPipelining, bool enablePrefetchTiling, Logger log)
            : _enableVerticalFusionPipelining(enableVerticalFusionPipelining),
              _enablePrefetchTiling(enablePrefetchTiling) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _enableVerticalFusionPipelining = false;
    bool _enablePrefetchTiling = true;
};

mlir::LogicalResult MergeVfSubgraphsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (enableVerticalFusionPipelining.hasValue()) {
        _log.trace("Overloading MergeVfSubgraphsPass argument by MLIR variable");
        _enableVerticalFusionPipelining = enableVerticalFusionPipelining;
    }
    if (tilingMode.hasValue()) {
        _log.trace("Overloading the default value {0} of the '_enablePrefetchTiling' field to the value {1} of the "
                   "pass option 'tilingMode' generated by MLIR",
                   _enablePrefetchTiling, tilingMode.getValue());
        _enablePrefetchTiling = tilingMode.getValue() == "PREFETCH";
    }
    return mlir::success();
}

//
// safeRunOnModule
//

void MergeVfSubgraphsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    const auto costFunction = std::make_unique<VPU::LayerVPUNNCost>(func);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MergeVFRegionRewriter>(&ctx, _enableVerticalFusionPipelining, _enablePrefetchTiling, costFunction,
                                        _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createMergeVfSubgraphsPass
//

std::unique_ptr<mlir::Pass> VPU::createMergeVfSubgraphsPass(bool enableVerticalFusionPipelining,
                                                            bool enablePrefetchTiling, Logger log) {
    return std::make_unique<MergeVfSubgraphsPass>(enableVerticalFusionPipelining, enablePrefetchTiling, log);
}
