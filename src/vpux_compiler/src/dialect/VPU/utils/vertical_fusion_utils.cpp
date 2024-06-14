//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/vf_axis_increment.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"

#include "llvm/ADT/SetOperations.h"

using namespace vpux;
using namespace VPU;

TilingStorage vpux::VPU::restoreTilingRegions(VPU::VerticalFusionOp vfOp, Logger log,
                                              const TilingOperationStorage::UPtr& opStorage) {
    auto storage = calculateTilingRegions(
            vfOp, ArrayRef(parseIntArrayAttr<int64_t>(vfOp.getTilingStrategy().cast<mlir::ArrayAttr>())), log,
            opStorage);

    VPUX_THROW_WHEN(mlir::failed(storage), "Restored tiling {0} of operation {1} is incorrect",
                    vfOp.getTilingStrategy(), vfOp);

    return storage.value();
}

mlir::FailureOr<TilingStorage> vpux::VPU::calculateTilingRegions(VPU::VerticalFusionOp vfOp, const OutputTiling& tiles,
                                                                 Logger log,
                                                                 const TilingOperationStorage::UPtr& opStorage) {
    auto termination = vfOp.getBody()->getTerminator();

    if (termination == nullptr) {
        return mlir::failure();
    }

    if (termination->getNumOperands() == 0) {
        return mlir::failure();
    }

    auto lastOp = termination->getOperands().back().getDefiningOp();

    if (lastOp == nullptr) {
        return mlir::failure();
    }

    return calculateTilingRegions(lastOp, tiles, log, opStorage);
}

mlir::FailureOr<TilingStorage> vpux::VPU::calculateTilingRegions(mlir::Operation* operation, const OutputTiling& tiles,
                                                                 Logger log,
                                                                 const TilingOperationStorage::UPtr& opStorage,
                                                                 std::optional<size_t> numTile) {
    TilingStorage storage;

    if (auto tilingInfoInterface = mlir::dyn_cast<VPU::TilingInfoOpInterface>(operation)) {
        try {
            if (!isMultiClusterCompatibleForTiling(operation, tiles, log) ||
                !tilingInfoInterface.isSupportedTiling(tiles, TilingMode::ISOLATED, log)) {
                return mlir::failure();
            }
        } catch (Exception&) {
            return mlir::failure();
        }
    }

    for (const auto& item : tiles | indexed) {
        auto tile = item.value();

        auto inputTiling = TilingInfo(ArrayRef({tile}));
        if (auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(operation)) {
            inputTiling = tilingBuilderOp.backInferTileInfo(tile, log);
        } else if (auto tilingViewLikeOp = mlir::dyn_cast<VPU::TilingViewLikeOpInterface>(operation)) {
            inputTiling = tilingViewLikeOp.backInferTileInfo(tile, log);
        } else {
            VPUX_THROW("Unsupported operation type {0} for VF", operation->getName());
        }

        const auto tileNumber = numTile.value_or(item.index());

        if (opStorage != nullptr) {
            opStorage->insert(operation, tileNumber, std::make_pair(inputTiling, tile));
            log.trace("TileInfo inserted for operation {0} tile {1}, {2}", *operation, tileNumber, tile);
        }

        for (const auto& op : operation->getOperands() | indexed) {
            const auto operand = op.value();
            const auto indexOp = op.index();

            if (auto arg = operand.dyn_cast<mlir::BlockArgument>()) {
                storage.insert(arg.getArgNumber(), tileNumber, inputTiling.tiles[indexOp]);
                log.trace("TileInfo inserted for argument {0} tile {1}, {2}", arg.getArgNumber(), tileNumber,
                          inputTiling.tiles[indexOp]);
                continue;
            }
            const auto oneTile = {inputTiling.tiles[indexOp]};
            auto innerStorage = calculateTilingRegions(operand.getDefiningOp(), oneTile, log, opStorage,
                                                       numTile.value_or(item.index()));

            if (mlir::failed(innerStorage)) {
                return mlir::failure();
            }

            storage.merge(innerStorage.value());
        }
    }

    return storage;
}

mlir::FailureOr<TilingStorage> vpux::VPU::calculateTilingRegions(VPU::VerticalFusionOp vfOp,
                                                                 ArrayRef<int64_t> tilingStrategy, Logger log,
                                                                 const TilingOperationStorage::UPtr& opStorage) {
    const auto outputShape = getShape(vfOp->getResult(0));
    const auto strategy = Shape(tilingStrategy);

    const auto tiles = fillDividedTiles(vfOp, strategy, outputShape);
    if (mlir::failed(tiles)) {
        return mlir::failure();
    }

    return calculateTilingRegions(vfOp, tiles.value(), log, opStorage);
}

int64_t vpux::VPU::getTilingLimit(Dim axis, ArrayRef<mlir::Operation*> operations) {
    const auto axisLengths =
            to_small_vector(operations | transformed([&](auto* op) {
                                auto limit = getMaxNumTiles(op)[axis.ind()];
                                if (axis.ind() >= Dims4D::Act::getSpatialDim(0).ind()) {
                                    limit /= MINIMUM_LENGTH_TILING;
                                }

                                limit = std::min(limit, VPU::NCEInvariant::VPU_DIMENSION_LIMIT / MINIMUM_LENGTH_TILING);
                                return limit;
                            }));

    auto axisIncrement = getVFAxisIncrement(axis);
    VPUX_THROW_WHEN(axisIncrement == nullptr, "Cannot get functions to get values for axis {0}", axis);

    return axisIncrement->getLimitValue(axisLengths);
}

// get a valid tiling strategy for VF block between the given range of tiling strategy
// it returns mlir::failure() if all tiling strategies in this range can't be supported by all operations or operations
// can't fit in CMX
// otherwise, return the valid strategy that is close to the lower or upper boundary according to closeToUpperLimit
// parameter
mlir::FailureOr<SmallVector<int64_t>> getValidTilingStrategyFromRange(
        VPU::VerticalFusionOp op, ArrayRef<int64_t> lowerTilingStrategy, ArrayRef<int64_t> upperTilingStrategy,
        bool closeToUpperLimit, Dim tilingAxis, TilingOperationStorage::UPtr& opStorage, Logger log) {
    SmallVector<int64_t> validTilingStrategy =
            closeToUpperLimit ? to_small_vector(upperTilingStrategy) : to_small_vector(lowerTilingStrategy);

    auto notBeyondBoundary = [](int64_t value, int64_t lowerLimit, int64_t upperLimit, bool closeToUpperLimit) {
        return closeToUpperLimit ? value >= lowerLimit : value <= upperLimit;
    };

    auto axisIncrement = getVFAxisIncrement(tilingAxis);
    VPUX_THROW_WHEN(axisIncrement == nullptr, "Cannot get functions to get values for axis {0}", tilingAxis);

    while (notBeyondBoundary(validTilingStrategy[tilingAxis.ind()], lowerTilingStrategy[tilingAxis.ind()],
                             upperTilingStrategy[tilingAxis.ind()], closeToUpperLimit)) {
        auto curOpStorage = std::make_unique<TilingOperationStorage>();
        auto tilingRegions = calculateTilingRegions(op, validTilingStrategy, log, curOpStorage);
        if (!mlir::failed(tilingRegions)) {
            // a valid strategy is found
            opStorage.reset(curOpStorage.release());
            return validTilingStrategy;
        }

        auto currentValue = validTilingStrategy[tilingAxis.ind()];

        if (closeToUpperLimit) {
            axisIncrement->decreasedValue(validTilingStrategy[tilingAxis.ind()], lowerTilingStrategy[tilingAxis.ind()]);
        } else {
            axisIncrement->increasedValue(validTilingStrategy[tilingAxis.ind()], upperTilingStrategy[tilingAxis.ind()]);
        }

        if (currentValue == validTilingStrategy[tilingAxis.ind()]) {
            return mlir::failure();
        }
    }

    // no valid strategy can be found
    return mlir::failure();
}

// get a maximal valid tiling strategy for VF block between the given range of tiling strategy
// it returns mlir::failure() if all tiling strategies in this range can't be supported by all operations or operations
// can't fit in CMX
mlir::FailureOr<SmallVector<int64_t>> vpux::VPU::getMaximalValidTilingStrategyFromRange(
        VPU::VerticalFusionOp op, ArrayRef<int64_t> lowerTilingStrategy, ArrayRef<int64_t> upperTilingStrategy,
        Dim tilingAxis, TilingOperationStorage::UPtr& opStorage, Logger log) {
    return getValidTilingStrategyFromRange(op, lowerTilingStrategy, upperTilingStrategy, true, tilingAxis, opStorage,
                                           log);
}

// get a minimal valid tiling strategy for VF block between the given range of tiling strategy
// it returns mlir::failure() if all tiling strategies in this range can't be supported by all operations or operations
// can't fit in CMX
mlir::FailureOr<SmallVector<int64_t>> vpux::VPU::getMinimalValidTilingStrategyFromRange(
        VPU::VerticalFusionOp op, ArrayRef<int64_t> lowerTilingStrategy, ArrayRef<int64_t> upperTilingStrategy,
        Dim tilingAxis, TilingOperationStorage::UPtr& opStorage, Logger log) {
    return getValidTilingStrategyFromRange(op, lowerTilingStrategy, upperTilingStrategy, false, tilingAxis, opStorage,
                                           log);
}

std::optional<Dim> vpux::VPU::getVFTilingDim(ArrayRef<int64_t> tilingStrategy) {
    auto maxTiledLen = std::max_element(tilingStrategy.begin(), tilingStrategy.end());
    if (maxTiledLen != tilingStrategy.end() && *maxTiledLen != 1) {
        return Dim(std::distance(tilingStrategy.begin(), maxTiledLen));
    }
    return std::nullopt;
}

mlir::FailureOr<Dim> vpux::VPU::getVFTilingDim(ArrayRef<int64_t> tilingStrategy,
                                               ArrayRef<mlir::Operation*> operations) {
    auto dim = getVFTilingDim(tilingStrategy);
    if (dim.has_value()) {
        return dim.value();
    }

    auto allowedDims = getAllowedDims(operations, Logger::global());
    if (allowedDims.empty()) {
        return mlir::failure();
    }

    return allowedDims.front();
}

DimArr vpux::VPU::getAllowedDims(ArrayRef<mlir::Operation*> operations, Logger log) {
    auto order = DimsOrder::NHWC;
    const auto dimComparator = [&](Dim lhs, Dim rhs) -> bool {
        return order.hasDim(lhs) && order.hasDim(rhs) && order.dimPos(lhs) < order.dimPos(rhs);
    };

    DimArr allowedDims = order.toPermutation();
    for (auto tiledOperation : operations) {
        auto currentTiling = getTileDimOrder(tiledOperation, TilingMode::ISOLATED, log);
        llvm::sort(currentTiling, dimComparator);
        DimArr intersect;
        for (auto dim : currentTiling) {
            if (llvm::find(allowedDims, dim) != allowedDims.end()) {
                intersect.push_back(dim);
            }
        }
        allowedDims = std::move(intersect);
    }

    return allowedDims;
}

// calculate limit for number of tiles that can be supported by all operations in the VF block and all operations can
// fit into CMX with it
mlir::FailureOr<int64_t> vpux::VPU::getValidTilingLimit(VPU::VerticalFusionOp op, const Dim tilingAxis, Logger log) {
    auto tilingStrategy = parseIntArrayAttr<int64_t>(op.getTilingStrategy().cast<mlir::ArrayAttr>());
    const auto getOpPointer = [](auto& op) -> mlir::Operation* {
        return &op;
    };
    auto tilingLimit =
            getTilingLimit(tilingAxis, to_small_vector(op.getBody()->without_terminator() | transformed(getOpPointer)));

    SmallVector<int64_t> tilingMaxStrategy(tilingStrategy.size(), 1);
    tilingMaxStrategy[tilingAxis.ind()] = tilingLimit;

    // tilingMaxStrategy may be not valid for all operations in VF block here, needs to be legalized
    auto opStorage = std::make_unique<TilingOperationStorage>();
    auto validTilingMaxStrategy =
            getMaximalValidTilingStrategyFromRange(op, tilingStrategy, tilingMaxStrategy, tilingAxis, opStorage, log);
    if (mlir::failed(validTilingMaxStrategy)) {
        return mlir::failure();
    }

    return validTilingMaxStrategy.value()[tilingAxis.ind()];
}

VPU::VerticalFusionOp vpux::VPU::fuseOpsInBlock(mlir::PatternRewriter& rewriter, VPU::VerticalFusionOp vfOp,
                                                mlir::Operation* prevOp, mlir::ArrayAttr tilingInfo /*nullptr*/) {
    SmallVector<mlir::Operation*> prevOperations;
    auto prevOperands = prevOp->getOperands();
    SmallVector<mlir::Value> prevBlockArgs = prevOp->getOperands();
    mlir::Operation* lastOp = prevOp;
    const auto getOpPointer = [](auto& op) -> mlir::Operation* {
        return &op;
    };
    if (auto prevBlock = mlir::dyn_cast<VPU::VerticalFusionOp>(prevOp)) {
        prevBlockArgs.clear();
        llvm::copy(prevBlock.getBody()->getOperations() | transformed(getOpPointer),
                   std::back_inserter(prevOperations));
        llvm::copy(prevBlock.getBody()->getArguments(), std::back_inserter(prevBlockArgs));
        lastOp = prevBlock.getBody()->getTerminator()->getOperands().back().getDefiningOp();
    } else {
        prevOperations.push_back(prevOp);
    }

    SmallVector<size_t> argNumLastOp;
    SmallVector<size_t> argNumCurrentOp;
    mlir::DenseMap<size_t, size_t> opArgMapper;
    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange blockArgs) {
        mlir::IRMapping mapper;

        const auto curBlockArgs = vfOp.getBody()->getArguments();

        // map new operands with previous ones for both blocks
        for (size_t i = 0; i < blockArgs.size(); ++i) {
            if (i < prevBlockArgs.size()) {
                // map operands of first block with current ones
                mapper.map(prevBlockArgs[i], blockArgs[i]);

                // in case there is operand in second block which also
                // can be mapped with this operands - map them too
                if (opArgMapper.count(i) != 0) {
                    mapper.map(curBlockArgs[opArgMapper[i]], blockArgs[i]);
                }
            } else {
                // map other operands
                if (argNumCurrentOp.size() > i - prevBlockArgs.size() &&
                    curBlockArgs.size() > argNumCurrentOp[i - prevBlockArgs.size()]) {
                    mapper.map(curBlockArgs[argNumCurrentOp[i - prevBlockArgs.size()]], blockArgs[i]);
                }
                if (opArgMapper.count(i) != 0) {
                    mapper.map(curBlockArgs[opArgMapper[i]], blockArgs[i]);
                }
            }
        }

        SmallVector<mlir::Value> newResults;

        const auto copyOps = [&](auto operations) {
            for (auto* op : operations) {
                if (!mlir::isa<VPU::YieldOp>(op)) {
                    auto* clonedOp = builder.clone(*op, mapper);
                    if (op == lastOp && !argNumLastOp.empty()) {
                        for (auto index : argNumLastOp) {
                            mapper.map(curBlockArgs[index], clonedOp->getResult(0));
                        }
                    }
                } else {
                    for (auto operand : op->getOperands()) {
                        if (operand.getDefiningOp() != lastOp) {
                            newResults.push_back(mapper.lookupOrDefault(operand));
                        }
                    }
                }
            }
        };

        copyOps(prevOperations);
        copyOps(vfOp.getBody()->getOperations() | transformed(getOpPointer));

        builder.create<VPU::YieldOp>(loc, newResults.back());
    };

    SmallVector<mlir::Value> newOperands(prevOperands.begin(), prevOperands.end());

    VPUX_THROW_WHEN(lastOp == nullptr, "Couldn't find last operation in region {0}", prevOp);

    // for all operands in current region
    // sort them in following baskets
    // argNumLastOp - if operand is previous region
    // argNumCurrentOp - arguments of current region
    // opArgMapper - in case operand is already in the list,
    // map this operand and argument of current block in order to
    // create right correlation
    for (auto arg : vfOp.getBody()->getArguments()) {
        auto operand = vfOp.getOperand(arg.getArgNumber());
        if (operand.getDefiningOp() == prevOp) {
            argNumLastOp.push_back(arg.getArgNumber());
        } else {
            const auto value = llvm::find(newOperands, operand);
            if (value == newOperands.end()) {
                newOperands.push_back(operand);
                argNumCurrentOp.push_back(arg.getArgNumber());
            } else {
                opArgMapper[std::distance(newOperands.begin(), value)] = arg.getArgNumber();
            }
        }
    }

    if (tilingInfo == nullptr) {
        tilingInfo = vfOp.getTilingStrategy();
    }

    return rewriter.create<VPU::VerticalFusionOp>(vfOp.getLoc(), vfOp->getResultTypes(), newOperands, bodyBuilder,
                                                  tilingInfo);
}

VPUNNCostParameters fillInCostParam(mlir::Operation* operation, const OutputTiling& tiling,
                                    const SmallVector<TileInfo>& inputTiles, const bool enablePrefetching, Logger log) {
    auto mcStrategy = VPU::MultiClusterStrategy::Clustering;
    if (auto mcOperation = mlir::dyn_cast<VPU::ClusteredOpInterface>(operation)) {
        mcStrategy = mcOperation.getMultiClusterStrategy().value_or(mcStrategy);
    }

    auto mode = TilingMode::ISOLATED;
    if (auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(operation)) {
        // cache it in the storage E-121580
        mode = getTilingSupportedMode(tilingBuilder, enablePrefetching, log);
    }
    SmallVector<OutputTiling> inputAllTiles;
    if (!inputTiles.empty()) {
        inputAllTiles.push_back(inputTiles);
    }
    return VPUNNCostParameters(mcStrategy, tiling, mode, inputAllTiles);
}

VPUNNCostParameters fillInCostParam(mlir::Operation* operation,
                                    const std::unique_ptr<TilingOperationStorage>& opStorage, size_t index, Logger log,
                                    bool enablePrefetching) {
    auto inputOutputTiling = opStorage->get(operation, index);

    VPUX_THROW_WHEN(!inputOutputTiling.has_value(), "Couldn't find tile information for operation {0} and tile {1}",
                    *operation, index);
    const auto inputOutputTilingPair = inputOutputTiling.value();

    return fillInCostParam(operation, {inputOutputTilingPair.second}, inputOutputTilingPair.first.tiles,
                           enablePrefetching, log);
}

void existedVFSpilling(mlir::Operation* operation, const std::unique_ptr<TilingOperationStorage>& opStorage,
                       SmallVector<mlir::Operation*>& users, size_t index) {
    auto inputTiling = opStorage->get(operation, index);

    VPUX_THROW_WHEN(!inputTiling.has_value(), "Couldn't find tile information for operation {0} and tile {1}",
                    operation, index);
    const auto inputTilingPair = inputTiling.value();

    auto outputOriginTiling = inputTilingPair.second;
    llvm::copy_if(operation->getResult(0).getUsers(), std::back_inserter(users), [&](auto* user) {
        auto vfTilingInfo = opStorage->get(user, index);

        if (!vfTilingInfo.has_value()) {
            return false;
        }
        const auto vfTiles = vfTilingInfo.value().first.tiles;

        return llvm::find(vfTiles, outputOriginTiling) == vfTiles.end();
    });
}

StrategyCost getVFCostPipelined(const int tilesNumber, VFConfig& config,
                                const std::unique_ptr<TilingOperationStorage>& opStorage,
                                const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log) {
    StrategyCost pipelinedCost = 0;
    // create a structure which reflects the execution order of the IR
    // same way as scheduler does

    auto pipelinedStructure = VFContainerPipelineStorage();

    // first "tile" is fully filled as it is,
    // next tile might be pipelined
    auto& operations = config.getVFOperations();

    // everything will be in one container with index 0 to simplify
    int containerIndex = 0;

    // storage size is flexible. minimum number of "containers" is a number of operations
    auto storageSize = operations.size();
    std::map<size_t, size_t> opStorageMapping;
    for (auto index : irange(tilesNumber)) {
        // first row should be presented in the container
        // DPU -> SW -> DPU
        // next line has to be pipelined
        // DPU -> SW  -> DPU
        //        DPU -> SW  -> DPU
        // E#95184 for extending the case
        if (index % 2 == 0) {
            for (auto opIndex : irange(operations.size())) {
                pipelinedStructure.insert(
                        opIndex, containerIndex,
                        VFPipelineContainer(operations[opIndex], fillInCostParam(operations[opIndex], opStorage, index,
                                                                                 log, config.isPipelined())));
            }
            ++containerIndex;
            continue;
        }

        for (auto opIndex : irange(operations.size())) {
            auto tilingInfo = fillInCostParam(operations[opIndex], opStorage, index, log, config.isPipelined());
            auto foundMapping = opStorageMapping.find(opIndex);
            auto containerNumber = opIndex;
            if (foundMapping != opStorageMapping.end()) {
                containerNumber = foundMapping->second;
            } else {
                ++containerNumber;
                do {
                    auto currentPipeline = pipelinedStructure.get(containerNumber, containerIndex - 1);
                    if (!currentPipeline.has_value() || currentPipeline.value().hasOperType(operations[opIndex])) {
                        break;
                    }
                    ++containerNumber;
                } while (containerNumber < storageSize);
                opStorageMapping[opIndex] = containerNumber;
            }

            auto pipeline = pipelinedStructure.get(containerNumber, containerIndex);
            if (!pipeline.has_value()) {
                pipelinedStructure.insert(containerNumber, containerIndex,
                                          VFPipelineContainer(operations[opIndex], tilingInfo));
                ++storageSize;
            } else {
                pipeline.value().addOperation(operations[opIndex], tilingInfo);
            }
        }
    }

    // iterate through the structure, accumulate the cost for each "container", taking maximum from it.
    for (auto opIndex : irange(storageSize)) {
        StrategyCost cost = 0;
        for (auto container : pipelinedStructure.gatherValue(opIndex)) {
            cost = std::max(cost, container.maxCost(costFunction));
        }
        pipelinedCost += cost;
    }

    return pipelinedCost;
}

StrategyCost vpux::VPU::getVFCost(const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, VPU::VerticalFusionOp vfOp,
                                  Logger log, bool prefetching /*true*/,
                                  mlir::ArrayAttr tilingStrategyAttr /* nullptr */) {
    StrategyCost fullCost = 0;
    if (tilingStrategyAttr == nullptr) {
        tilingStrategyAttr = vfOp.getTilingStrategy();
    }

    auto tilingDims = parseIntArrayAttr<int64_t>(tilingStrategyAttr);
    VFConfig vfConfig(vfOp, prefetching);
    auto operations = vfConfig.getVFOperations();

    const auto dim = getVFTilingDim(tilingDims);

    if (operations.size() == 1) {
        auto* operation = operations.front();
        OutputTiling tiles;
        if (dim.has_value()) {
            auto tiling = fillDividedTiles(operation, Shape(tilingDims), getShape(operation->getResult(0)));
            VPUX_THROW_WHEN(mlir::failed(tiling), "Incorrect tiling {0} for vf {1}", tilingDims, vfOp);
            tiles = tiling.value();
        }

        const auto costParameters = fillInCostParam(operation, tiles, {}, prefetching, log);
        return costFunction->getStrategyCost(operation, costParameters);
    }

    auto operationStorage = std::make_unique<TilingOperationStorage>();
    auto tilingStorage = calculateTilingRegions(vfOp, tilingDims, log, operationStorage);

    VPUX_THROW_WHEN(mlir::failed(tilingStorage), "Incorrect tiles for vf {0}", vfOp);

    const auto tileNum = dim.has_value() ? tilingDims[dim.value().ind()] : 1;

    // we may get tiling strategy and merge VF for pipeline candidate without VF pipelining enabled
    // validate CMX size to check if it's a real pipelining case, so that we can get accurate VF cost
    if (vfConfig.isPipelined() && validateCMXSize(vfConfig, operationStorage, log)) {
        VPUX_THROW_UNLESS(tileNum > 1, "Subgraph cannot be pipelined with {0} tiles", tileNum);
        return getVFCostPipelined(tileNum, vfConfig, operationStorage, costFunction, log);
    }

    // sizes of dmas from cmx
    for (auto* operation : operations) {
        for (auto operand : operation->getOperands()) {
            if (auto arg = operand.dyn_cast<mlir::BlockArgument>()) {
                if (vfOp.getOperand(arg.getArgNumber()).getDefiningOp<Const::DeclareOp>()) {
                    continue;
                }
                auto findOperand = [&](mlir::Value value) -> bool {
                    return value == arg;
                };
                auto parentOp = vfOp.getOperand(arg.getArgNumber()).getDefiningOp();
                if (parentOp && parentOp->hasAttr(tilingStrategy)) {
                    const auto strategy = Shape(
                            parseIntArrayAttr<int64_t>(parentOp->getAttr(tilingStrategy).cast<mlir::ArrayAttr>()));
                    if (llvm::none_of(strategy,
                                      [&](auto value) {
                                          return value > 1;
                                      }) &&
                        !vfConfig.isPotentiallyPipelined()) {
                        const auto userCostParam = fillInCostParam(operation, operationStorage, 0, log, prefetching);
                        fullCost += costFunction->getSpillingReadCost(operation, userCostParam, parentOp, findOperand);
                    }
                }
            }
        }
    }

    for (auto index : irange(tileNum)) {
        for (auto* op : operations) {
            const auto costParameters = fillInCostParam(op, operationStorage, index, log, prefetching);

            // isolated operation cost
            fullCost += costFunction->getStrategyCost(op, costParameters);

            SmallVector<mlir::Operation*> spillUsers;
            existedVFSpilling(op, operationStorage, spillUsers, index);

            if (!spillUsers.empty()) {
                fullCost += costFunction->getSpillingWriteCost(op, costParameters);
                for (auto* user : spillUsers) {
                    const auto userCostParam = fillInCostParam(user, operationStorage, index, log, prefetching);
                    fullCost += costFunction->getSpillingReadCost(user, userCostParam, op);
                }
            }
        }
    }

    return fullCost;
}

bool vpux::VPU::validateCMXSize(VFConfig& config, const TilingOperationStorage::UPtr& opStorage, Logger log,
                                Byte reservedMemory /*Byte(0)*/) {
    auto* largest = config.getLargestOp();
    // assuming almost all tiles are same
    const auto index = 0;
    auto inputSize = Byte(0);

    for (auto op : config.getInputs()) {
        auto tileInfo = opStorage->get(op, index);
        VPUX_THROW_WHEN(!tileInfo.has_value(), "There is no information about tile {0} of operation {1}", index, *op);

        auto tileTypes = getTileTypes(op, tileInfo.value().second, tileInfo.value().first);
        VPUX_THROW_WHEN(tileTypes.empty(), "There are not enough types for tile of operation {0}", *op);
        // exclude output type information
        tileTypes.pop_back();
        for (auto type : tileTypes) {
            inputSize += type.getTotalAllocSize();
        }
    }

    auto outputSize = Byte(0);

    for (auto op : config.getOutputs()) {
        auto tileInfo = opStorage->get(op, index);
        VPUX_THROW_WHEN(!tileInfo.has_value(), "There is no information about tile {0} of operation {1}", index, *op);

        auto tileTypes = getTileTypes(op, tileInfo.value().second, tileInfo.value().first);
        VPUX_THROW_WHEN(tileTypes.empty(), "There is no output type for tile of operation {0}", *op);

        auto type = tileTypes.back();
        outputSize += type.getTotalAllocSize();
    }

    auto opTiling = opStorage->get(largest, index);
    VPUX_THROW_WHEN(!opTiling.has_value(), "There is no information about tile {0} of operation {1}", index, *largest);
    log.trace("Check for tile number {0}: inputs' size {1} outputs's size {2}", index, inputSize, outputSize);
    const auto thresholdCMXSize = config.isPipelined() ? getTotalCMXVFPipelineFragmentationAwareSize(largest)
                                                       : getTotalCMXFragmentationAwareSize(largest);
    if (inputSize + outputSize + reservedMemory +
                VPU::getRequiredCMX(largest, opTiling.value().second, log, opTiling.value().first) >=
        thresholdCMXSize) {
        return false;
    }

    return true;
}
