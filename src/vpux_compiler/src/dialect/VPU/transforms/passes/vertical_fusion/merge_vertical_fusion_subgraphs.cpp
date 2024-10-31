//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/type_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
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

// set VF users for correct tensor size calculation
class VFSubgraphUserSetter {
public:
    VFSubgraphUserSetter(VerticalFusionOp original, VerticalFusionOp candidate)
            : _mOriginalSubgraph(original), _mCandidateSubgraph(candidate) {
        moveUsers(_mOriginalSubgraph, _mCandidateSubgraph);
    }
    ~VFSubgraphUserSetter() {
        moveUsers(_mCandidateSubgraph, _mOriginalSubgraph);
    }

private:
    void moveUsers(VerticalFusionOp from, VerticalFusionOp to);

    VerticalFusionOp _mOriginalSubgraph;
    VerticalFusionOp _mCandidateSubgraph;
};

void VFSubgraphUserSetter::moveUsers(VerticalFusionOp from, VerticalFusionOp to) {
    from.getResult(0).replaceAllUsesWith(to.getResult(0));
}

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
                             VPU::VerticalFusionOp mergedVFOp, mlir::ArrayAttr tiling,
                             std::optional<StrategyCost> mergedVFCost) const;
    int64_t getMaxDimLimit(const Dim axis, ArrayRef<mlir::Operation*> operation) const;
    bool getOptimalTilingStrategy(SmallVector<int64_t>& tilingArray, const Dim dim, const int64_t minTiles,
                                  const int64_t maxTiles, VFConfig& config) const;
    bool waitOtherUsers(VPU::VerticalFusionOp newBlock, VPU::VerticalFusionOp parentVFOp) const;
    mlir::ArrayAttr getVFTilingInfo(VPU::VerticalFusionOp newBlock, VPU::VerticalFusionOp parentVFOp,
                                    VPU::VerticalFusionOp mergedVFOp, std::optional<StrategyCost>& mergedCost) const;
    bool alignMCTiling(VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp) const;
    mlir::FailureOr<Dim> getTilingAxis(SmallVector<int64_t>& tilingStrategy, VPU::VerticalFusionOp prevOp,
                                       VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp mergedVFOp,
                                       std::optional<StrategyCost>& mergedCost) const;

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

    const auto getCurrInputArgument = [](VPU::VerticalFusionOp currentOp,
                                         VPU::VerticalFusionOp prevOp) -> mlir::BlockArgument {
        for (auto blockArg : currentOp.getBody()->getArguments()) {
            auto operand = currentOp.getOperand(blockArg.getArgNumber());
            if (operand.getDefiningOp() == prevOp.getOperation()) {
                return blockArg;
            }
        }
        return nullptr;
    };

    // Get output op of previous vf region
    auto prevOutputOp = prevOp.getBody()->getTerminator()->getOperands().back().getDefiningOp();
    // Get input arg of current vf region corresponding to previous vf op
    auto currInputArg = getCurrInputArgument(currentOp, prevOp);
    VPUX_THROW_UNLESS(currInputArg != nullptr,
                      "No corresponding input argument found for current VF region with previous VF region");

    const auto isClusteredOpWithMCStrategy = [](mlir::Operation* op) {
        auto clusterOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(op);
        return clusterOp != nullptr && clusterOp.getMultiClusterStrategy().has_value();
    };

    const auto getOutputDistributedType = [](VPU::ClusteredOpInterface clusteredOp) {
        const auto outputType = clusteredOp->getResult(0).getType().cast<NDTypeInterface>();
        const auto numClusters =
                clusteredOp.getOptimalNumClusters(outputType.getShape(), clusteredOp.getMultiClusterStrategy().value());

        auto ndType = VPU::getDistributedOutputTypeFromOp(clusteredOp, outputType, numClusters).cast<NDTypeInterface>();
        if (auto sparseTensorType = ndType.dyn_cast<VPU::SparseTensorType>()) {
            ndType = sparseTensorType.getData().cast<NDTypeInterface>();
        }

        return ndType.dyn_cast_or_null<VPU::DistributedTensorType>();
    };

    const auto getInputDistributedType = [](VPU::ClusteredOpInterface clusteredOp, mlir::Value inputOperand) {
        const auto inputType = inputOperand.getType().cast<NDTypeInterface>();
        const auto numClusters =
                clusteredOp.getOptimalNumClusters(inputType.getShape(), clusteredOp.getMultiClusterStrategy().value());

        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(clusteredOp.getOperation());

        auto ndType =
                (nceOp != nullptr && nceOp.getWeightsOperand() == inputOperand)
                        ? VPU::getDistributedFilterTypeFromOp(nceOp, inputType, numClusters).cast<NDTypeInterface>()
                        : VPU::getDistributedActivationTypeFromOp(clusteredOp, inputType, numClusters)
                                  .cast<NDTypeInterface>();
        if (auto sparseTensorType = ndType.dyn_cast<VPU::SparseTensorType>()) {
            ndType = sparseTensorType.getData().cast<NDTypeInterface>();
        }

        return ndType.dyn_cast_or_null<VPU::DistributedTensorType>();
    };

    const auto inferInputDistributedType = [](VPU::DistributedTensorType srcDistType,
                                              ArrayRef<mlir::Operation*> inputViewOps) {
        auto inputDistType = mlir::cast<vpux::NDTypeInterface>(srcDistType);
        auto distribution = VPU::DistributionInfo::getClassFromAttr(srcDistType.getDistribution());
        for (auto viewOp : inputViewOps) {
            if (auto distCastOp = mlir::dyn_cast<VPU::DistributedCastOpInterface>(viewOp)) {
                auto castedTypeWithDistribution =
                        distCastOp.inferCastedTypeAndDistribution(inputDistType, distribution);
                if (mlir::succeeded(castedTypeWithDistribution)) {
                    inputDistType = mlir::cast<vpux::NDTypeInterface>(castedTypeWithDistribution.value().first);
                    distribution = castedTypeWithDistribution.value().second;
                }
            }
        }
        TensorDistributionMap distributionMap;
        distributionMap.insert(std::make_pair(inputDistType, distribution));
        return mlir::cast<VPU::DistributedTensorType>(
                getDistributedTypeFromDistributionMap(inputDistType, distributionMap));
    };

    // Check if previous output op has MC strategy
    const auto isPrevOutOpWithMCStrategy = isClusteredOpWithMCStrategy(prevOutputOp);
    const auto prevOutDistType = isPrevOutOpWithMCStrategy
                                         ? getOutputDistributedType(mlir::cast<VPU::ClusteredOpInterface>(prevOutputOp))
                                         : nullptr;

    const auto hasTrueOverlappedParams = [](VPU::DistributedTensorType tensor) {
        if (tensor == nullptr) {
            return false;
        }
        if (tensor.getDistribution().getMode().getValue() != VPU::DistributionMode::OVERLAPPED) {
            return false;
        }
        if (tensor.getPerClusterMemoryShapes() != tensor.getPerClusterComputeShapes()) {
            return true;
        }
        if (tensor.getPerClusterMemoryShapeOffsets() != tensor.getPerClusterComputeShapeOffsets()) {
            return true;
        }
        return false;
    };
    bool outputTrueOverlapped = hasTrueOverlappedParams(prevOutDistType);
    bool isSWLayer = mlir::isa<VPU::SWOpInterface>(prevOutputOp);

    // Here we need to ensure either all current input ops and previous output op have no mc strategy,
    // or all have mc stratgy with compatible distributed tensor types
    for (auto currInputOp : currInputArg.getUsers()) {
        SmallVector<mlir::Operation*> currInputViewLikeOps;
        while (mlir::isa<VPU::TilingViewLikeOpInterface>(currInputOp) && currInputOp->hasOneUse()) {
            currInputViewLikeOps.push_back(currInputOp);
            currInputOp = *(currInputOp->getUsers().begin());
        }
        const auto isCurrInOpWithMCStrategy = isClusteredOpWithMCStrategy(currInputOp);
        if (isPrevOutOpWithMCStrategy != isCurrInOpWithMCStrategy) {
            return false;
        }
        if (isPrevOutOpWithMCStrategy && isCurrInOpWithMCStrategy) {
            auto currInputOperand = currInputViewLikeOps.empty() ? currInputArg.cast<mlir::Value>()
                                                                 : currInputViewLikeOps.back()->getResult(0);
            auto actualCurrInDistType =
                    getInputDistributedType(mlir::cast<VPU::ClusteredOpInterface>(currInputOp), currInputOperand);
            auto inferredCurrInDistType = inferInputDistributedType(prevOutDistType, currInputViewLikeOps);

            // TODO E#92130 extend Shave operations with OVERLAPPED param propagation
            if ((outputTrueOverlapped && mlir::isa<VPU::SWOpInterface>(currInputOp)) ||
                (isSWLayer && hasTrueOverlappedParams(actualCurrInDistType))) {
                return false;
            }

            if (areDistributionAttrsCompatible(inferredCurrInDistType, actualCurrInDistType, true).failed()) {
                return false;
            }
        }
    }

    return true;
}

// Get operandNumber for prevOp output in currentOp inputs
size_t getLinkNumber(VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp) {
    auto operands = currentOp->getOperands();
    auto operandIt = llvm::find_if(operands, [&](auto operand) {
        return operand.getDefiningOp() == prevOp;
    });
    VPUX_THROW_WHEN(operandIt == operands.end(),
                    "Cannot find the operand number for the operation {0} in the current block {1}", prevOp, currentOp);
    return std::distance(operands.begin(), operandIt);
}

/*
 Function checks if two blocks suit to be merged in one on following criterias:
 1. Number of operations doesn't exceed the limit
 2. In case there is only one operation in the block, it might be merged as first op in the block
 3. All multicluster strategies are same for both blocks if there are any
 4. Required CMX memory by constant weights shouldn't exceed the size of the whole memory
*/
bool MergeVFRegionRewriter::checkVFCostFunction(VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currentOp,
                                                VPU::VerticalFusionOp mergedVFOp, mlir::ArrayAttr tiling,
                                                std::optional<StrategyCost> mergedVFCost) const {
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

    const auto isSpillBetween = [&](auto& prevTilingStrategy, auto& currentTilingStrategy) {
        // There must be a spill when parent VF needs tiling but current VF doesn't
        if (llvm::any_of(prevTilingStrategy, moreThanOne) && !llvm::any_of(currentTilingStrategy, moreThanOne)) {
            const auto prevOutputSize = prevOp->getResult(0).getType().cast<NDTypeInterface>().getTotalAllocSize();
            if (prevOutputSize > VPU::getTotalCMXSize(prevOp)) {
                return true;
            }
        }

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

        if (isSpatialTiling(prevTilingStrategy)) {
            return true;
        }

        if (llvm::any_of(currentTilingStrategy, moreThanOne)) {
            if (isSpatialTiling(currentTilingStrategy)) {
                return true;
            }
            // A channel tiling case
            // Note: only care if current tiling strategy is moreThanOne, as the previous vertical fusion may have
            // cmx-concat optimization thus there's no spilling even though prevTilingStrategy is moreThanOne
            auto currentConfig = VFConfig(currentOp);
            auto currentOperations = currentConfig.getVFOperations();
            for (auto* op : currentOperations) {
                if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op)) {
                    if (nceOp.getWeightsOperand() != nullptr) {
                        return false;
                    }
                }
            }
            return true;
        }

        return false;
    };

    auto spillBetween = isSpillBetween(prevTilingStrategy, currentTilingStrategy);

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

        // Record the operation and its corresponding tiling strategy when back-infer subgraph
        std::unordered_map<mlir::Operation*, SmallVector<int64_t>> opTilingStrategyMap;
        backInferVFTilingStrategy(currentOp, currentTilingStrategy, opTilingStrategyMap);

        for (auto arg : currentVFOp->getArguments()) {
            auto operand = currentOp.getOperand(arg.getArgNumber());
            auto findOperand = [&](mlir::Value value) -> bool {
                return value == arg;
            };
            if (operand.getDefiningOp() == prevOp) {
                for (auto* user : arg.getUsers()) {
                    mcStrategy = getStrategy(user);
                    VPUX_THROW_WHEN(opTilingStrategyMap.find(user) == opTilingStrategyMap.end(),
                                    "Operation {0} is not in the map", user->getLoc());
                    const auto currentTiles =
                            fillDividedTiles(user, Shape(opTilingStrategyMap.at(user)), getShape(user->getResult(0)));

                    const auto userParams = VPUNNCostParameters(mcStrategy, currentTiles.value());
                    spillingCost += _vpunnCostFunction->getSpillingReadCost(user, userParams, prevOp.getOperation(),
                                                                            findOperand);
                }
            }
        }
    }

    // create new VF, in case the cost is worse, delete it
    // E-121586
    if (!mergedVFCost.has_value()) {
        VFSubgraphUserSetter setter(currentOp, mergedVFOp);
        mergedVFCost = getVFCost(_vpunnCostFunction, mergedVFOp, _log, _enablePrefetchTiling, tiling);

        VPUX_THROW_WHEN(!mergedVFCost.has_value(), "Couldn't calculate cost for {0}", mergedVFOp);
    }

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

            mergedVFCost.value() += types.back().getTotalAllocSize().count();
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
                        mergedVFCost.value() += 2 * operandCost.count();
                    }
                }
            }
        }
    }

    if (mergedVFCost.value() > prevCost + currentCost + spillingCost) {
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

        return mlir::succeeded(tilingRegions);
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

    if (mlir::failed(getValidStrategy)) {
        return false;
    }

    if (!validateCMXSize(config, opStorage, _log)) {
        // let VF with minimal requirements to be checked
        return !config.isPipelined();
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
            tilingArray[dim.ind()] = maxNTiles;
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
                                                          VPU::VerticalFusionOp mergedVFOp,
                                                          std::optional<StrategyCost>& mergedCost) const {
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

    // Record the operation and its corresponding tiling dim when back-infer subgraph
    std::unordered_map<mlir::Operation*, vpux::Dim> opDimMap;
    // Only for current VF Op check to skip restricted dims
    // E.g., VF{conv} -> VF{conv}, the first VF can support CTiling, but the second cannot
    const auto isRegionRestrictedDim = [](VPU::VerticalFusionOp currentOp,
                                          const std::unordered_map<mlir::Operation*, vpux::Dim>& opDimMap) {
        for (auto operation : currentOp.getBody()->getOps<VPU::VerticalFusionOpInterface>()) {
            auto restrictedAxes = operation.restrictedFusionAxes();
            VPUX_THROW_WHEN(opDimMap.find(operation.getOperation()) == opDimMap.end(),
                            "Operation {0} is not in the map", operation.getOperation()->getLoc());
            auto dim = opDimMap.at(operation.getOperation());
            if (llvm::find(restrictedAxes, dim) != restrictedAxes.end()) {
                return true;
            }
        }
        return false;
    };

    VFSubgraphUserSetter setter(currentOp, mergedVFOp);

    auto getVFTilingStrategy = [&](SmallVector<int64_t>& tilingArray, const Dim curDim, const Dim prevDim) -> bool {
        const auto maxTiles = getTilingLimit(curDim, vfConfig.getVFOperations());
        const auto minTiles = std::max(currentTiling[curDim.ind()], prevTiling[prevDim.ind()]);

        // get optimal tiling strategy for the 1st time
        auto result = getOptimalTilingStrategy(tilingArray, curDim, minTiles, maxTiles, vfConfig);
        if (result) {
            return tilingArray[curDim.ind()] != 1 || vfConfig.isPotentiallyPipelined();
        }

        // if it failed to get optimal stratey for the first time with VF pipelining
        // try to get strategy once again without VF pipelining
        // it might success becasue VF requires smaller CMX memory size without VF pipelining
        if (vfConfig.isPipelined()) {
            vfConfig.disableVFPipeline();
            result = getOptimalTilingStrategy(tilingArray, curDim, minTiles, maxTiles, vfConfig);
            vfConfig.restoreVFPipeline();

            // if we don't need tiling, VF is inefficient without pipelining
            return result && tilingArray[curDim.ind()] != 1;
        }

        return false;
    };

    const auto linkNumber = getLinkNumber(currentOp, prevOp);
    if (curHasTiling && prevHasTiling) {
        auto curInputAxes = backInferVFTilingDim(currentOp, curAxis.value(), opDimMap);
        if (curInputAxes[linkNumber] == prevAxis.value()) {
            // try to use current axis, otherwise try to find other axis
            if (!isRegionRestrictedDim(currentOp, opDimMap) &&
                getVFTilingStrategy(tilingArray, curAxis.value(), prevAxis.value())) {
                return curAxis.value();
            }
        }
    }

    DimArr allowedDims = getAllowedDims(vfConfig.getVFOperations(), _log);
    if (allowedDims.empty()) {
        return mlir::failure();
    }

    StrategyCost bestCost = std::numeric_limits<StrategyCost>::max();
    mlir::FailureOr<Dim> axis;
    for (auto dim : allowedDims) {
        // E.g., prevTiling [1, 3, 1, 1] -> permuteCast -> currentTiling [1, 1, 2, 1]
        // Thus we need dim backinfer to get correct axis to compare
        // As Vf inputs may be more than one, we need backinfer dim for each of them and use correct one
        auto curInputDims = backInferVFTilingDim(currentOp, dim, opDimMap);

        if (isRegionRestrictedDim(currentOp, opDimMap)) {
            continue;
        }

        // calculate optimal number of tiles for that dim
        SmallVector<int64_t> tilingAxisArray;
        if (!getVFTilingStrategy(tilingAxisArray, dim, curInputDims[linkNumber])) {
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

    if (mlir::succeeded(axis)) {
        mergedCost = bestCost;
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
                                                       VPU::VerticalFusionOp mergedVFOp,
                                                       std::optional<StrategyCost>& mergedCost) const {
    if (!alignMCTiling(currentOp, prevOp)) {
        return nullptr;
    }

    SmallVector<int64_t> tilingArray;
    const auto axis = getTilingAxis(tilingArray, prevOp, currentOp, mergedVFOp, mergedCost);

    if (mlir::failed(axis)) {
        return nullptr;
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
        std::optional<StrategyCost> mergedCost;
        tilingInfo = getVFTilingInfo(parentVFOp, vfOp, vfBlock, mergedCost);
        if (tilingInfo == nullptr) {
            rewriter.eraseOp(vfBlock);
            return mlir::failure();
        }

        if (!checkVFCostFunction(parentVFOp, vfOp, vfBlock, tilingInfo, mergedCost)) {
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
