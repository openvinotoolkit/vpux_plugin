//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/mc_strategy_getter.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/operation_strategies.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/strategy_opt_alg.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/strategy_state_provider.hpp"

#include <llvm/ADT/TypeSwitch.h>

namespace vpux::VPU {
namespace {

//
// StrategyManagerImplPass
//

class StrategyManagerImplPass final : public StrategyManagerImplBase<StrategyManagerImplPass> {
public:
    explicit StrategyManagerImplPass(bool enablePrefetchTiling, Logger log)
            : _enablePrefetchTiling(enablePrefetchTiling) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

    SmallVector<std::pair<Strategy, StrategyCost>> getOperationOptions(mlir::Operation* operation);
    SmallVector<VPU::MultiClusterStrategy> getAvailiableStrategies(ArchKind arch) const;
    bool checkDefaultStrategy(MultiClusterStrategy strategy) const;
    void exceptionalCases(SmallVector<VPU::MultiClusterStrategy>& strategies, mlir::Operation* operation) const;
    void fillInOptions(TilingOptions& options) const;
    bool mcTilingNeeded() const;

    std::shared_ptr<LayerVPUNNCost> _costModel;
    SmallVector<VPU::MultiClusterStrategy> _archStrategies;
    bool _enablePrefetchTiling = true;
    int64_t _numTiles;
};

void StrategyManagerImplPass::fillInOptions(TilingOptions& options) const {
    options.enablePrefetchTiling = _enablePrefetchTiling;
}

bool StrategyManagerImplPass::mcTilingNeeded() const {
    return _numTiles > 1;
}

mlir::LogicalResult StrategyManagerImplPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (tilingMode.hasValue()) {
        _log.trace("Overloading enablePrefetchTiling with an MLIR variable");
        _enablePrefetchTiling = tilingMode.getValue() == "PREFETCH";
    }
    return mlir::success();
}

bool StrategyManagerImplPass::checkDefaultStrategy(MultiClusterStrategy strategy) const {
    // check if strategy is default
    return strategy == MultiClusterStrategy::Clustering;
}

void StrategyManagerImplPass::exceptionalCases(SmallVector<VPU::MultiClusterStrategy>& strategies,
                                               mlir::Operation* operation) const {
    if (!mcTilingNeeded()) {
        return;
    }
    llvm::TypeSwitch<mlir::Operation*, void>(operation)

            .Case<NCEPermuteQuantizeOp>([&](NCEPermuteQuantizeOp) {
                strategies.push_back(MultiClusterStrategy::SplitOverWidth);
            });
}

SmallVector<std::pair<Strategy, StrategyCost>> StrategyManagerImplPass::getOperationOptions(
        mlir::Operation* operation) {
    SmallVector<std::pair<Strategy, StrategyCost>> strategies;
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(operation);

    auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(operation);

    if (clusteredOp == nullptr && tilingBuilderOp == nullptr) {
        return strategies;
    }

    auto operationStrategies = _archStrategies;
    exceptionalCases(operationStrategies, operation);

    for (auto strategy : operationStrategies) {
        if (clusteredOp != nullptr) {
            if (!(isStrategyCompatibleShape(clusteredOp,
                                            TileInfo(getShape(clusteredOp->getResult(0)), ShapeRef(), ShapeRef()),
                                            strategy, _log) &&
                  clusteredOp.checkStrategyCompatibility(strategy))) {
                continue;
            }
        } else if (!checkDefaultStrategy(strategy)) {
            continue;
        }

        bool hasPrefetch = false;
        do {
            mlir::ArrayAttr tilingStrategy;
            OutputTiling operationTiling;

            if (clusteredOp != nullptr) {
                clusteredOp.setMultiClusterStrategy(strategy);
            }

            auto mode = TilingMode::ISOLATED;
            bool prefetchTiling = hasPrefetch;
            // in case prefetch tiling is enabled, both cases should be taken into account
            // with prefetching and without
            hasPrefetch ^= _enablePrefetchTiling;
            const auto tilingNeeded = opNeedsTiling(operation, prefetchTiling, _log);
            if (tilingNeeded) {
                const auto layerTilingResult = getLayerTilingStrategy(tilingBuilderOp, prefetchTiling, mode, _log);

                if (mlir::failed(layerTilingResult)) {
                    continue;
                }

                operationTiling = layerTilingResult.value();
                VPUX_THROW_WHEN(operationTiling.empty(), "Couldn't get valid tiling for operation {0} in mode {1}",
                                operation->getLoc(), getTilingModeStr(mode));

                tilingStrategy = getIntArrayAttr(operation->getContext(), operationTiling[0].axis);
            } else if (clusteredOp != nullptr && !clusteredOp.doesLayerFitIntoCMX(strategy, Byte(0))) {
                _log.trace("Layer {0} doesn't fit in CMX with strategy {1}", operation->getLoc(), strategy);
                break;
            }

            if (operation->hasAttr(multiClusterStrategy)) {
                operation->removeAttr(multiClusterStrategy);
            }

            if (!tilingNeeded) {
                if (clusteredOp != nullptr && !clusteredOp.doesLayerFitIntoCMX(strategy, Byte(0))) {
                    _log.trace("Layer {0} doesn't fit in CMX with strategy {1}", operation->getLoc(), strategy);
                    continue;
                } else if (prefetchTiling) {
                    continue;
                }
            } else if (prefetchTiling && mode == TilingMode::ISOLATED) {
                continue;
            }

            const auto cost =
                    _costModel->getStrategyCost(operation, VPUNNCostParameters(strategy, operationTiling, mode));

            _log.trace("For operation {0} with MC strategy {1}-{2} vpunn returns cost {3}", operation->getLoc(),
                       strategy, tilingStrategy, cost);

            strategies.emplace_back(Strategy(strategy, tilingStrategy, mode), cost);

        } while (hasPrefetch);
    }

    return strategies;
}

SmallVector<VPU::MultiClusterStrategy> StrategyManagerImplPass::getAvailiableStrategies(ArchKind arch) const {
    auto mcListGetter = createMCStrategyGetter(arch, _numTiles);

    SmallVector<VPU::MultiClusterStrategy> strategies;
    mcListGetter->getMCStrategies(strategies);
    return strategies;
}

void StrategyManagerImplPass::safeRunOnFunc() {
    auto func = getOperation();
    auto module = func->getParentOfType<mlir::ModuleOp>();
    _costModel = std::make_shared<LayerVPUNNCost>(func);
    _numTiles = IE::getTileExecutor(module).getCount();
    _archStrategies = getAvailiableStrategies(VPU::getArch(module));

    // calculate cost for all possible strategies
    // assign strategy with min cost
    auto operationStrategies = std::make_shared<OperationStrategies>();

    const auto findStrategyCallback = [&](mlir::Operation* operation) {
        auto strategies = getOperationOptions(operation);

        if (strategies.empty()) {
            return;
        }

        for (auto& [strategy, cost] : strategies) {
            auto operationStrategy = std::make_pair(operation, strategy);
            operationStrategies->addStrategy(operationStrategy, cost);
        }

        // set current and best one
        auto maxCostIt = std::max_element(strategies.begin(), strategies.end(), [](auto lhs, auto rhs) {
            return lhs.second < rhs.second;
        });

        if (maxCostIt == strategies.end()) {
            return;
        }

        const auto maxCostStrategy = std::make_pair(operation, maxCostIt->first);
        operationStrategies->setCurrentStrategy(maxCostStrategy);
        operationStrategies->setBestStrategy(maxCostStrategy);
    };

    func.walk(findStrategyCallback);

    auto operations = operationStrategies->getAllOperations();
    if (operations.empty()) {
        return;
    }

    // optimization
    auto stateProvider = std::make_shared<DefaultStateProvider>(operationStrategies, _costModel);

    if (operations.size() > 1) {
        TilingOptions options;
        fillInOptions(options);
        const auto optAlgorithm = createAlgorithm(options, stateProvider, operationStrategies);
        optAlgorithm->optimize();
    } else {
        auto* operation = operationStrategies->getAllOperations().front();
        auto strategies = operationStrategies->getAllStrategies(operation);
        auto minCostIt = std::min_element(strategies.begin(), strategies.end(), [](auto lhs, auto rhs) {
            return lhs.strategyCost < rhs.strategyCost;
        });

        if (minCostIt == strategies.end()) {
            return;
        }

        const auto minCostStrategy = std::make_pair(operation, minCostIt->strategy);
        operationStrategies->setCurrentStrategy(minCostStrategy);
        operationStrategies->setBestStrategy(minCostStrategy);
    }

    // set best strategy to each operation
    const auto setStrategyCallback = [&](mlir::Operation* operation) {
        if (VPU::isPureViewOp(operation) || !operationStrategies->hasAnyStrategy(operation)) {
            return;
        }

        const auto bestResult = operationStrategies->getBestStrategy(operation);

        if (auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(operation)) {
            if (mcTilingNeeded()) {
                clusteredOp.setMultiClusterStrategy(bestResult.getMCStrategy());
            }
        }

        if (bestResult.getTilingStrategy() != nullptr) {
            if (auto tilingOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(operation)) {
                tilingOp->setAttr(tilingStrategy, bestResult.getTilingStrategy());
            }
        }
    };

    func.walk(setStrategyCallback);

    // E#81901 When assigning clustering strategy to ConvertOps for the following pattern(s) breaks the accuracy
    // ConvertOp            ConvertOp
    //     |                    |
    //     IntermediateOp       |
    //           |              |
    //              Layer (Layers with multiple inputs)
    // For such cases and strategies remove the strategy as a temporary measure
    const auto callbackConvert = [&](VPU::ConvertOp convertOp) {
        if (hasLayerWithMultipleInputs(convertOp)) {
            _log.info("SW Operation '{0}' {1} removed strategy {2}", convertOp->getName(), convertOp->getLoc(),
                      convertOp.getMultiClusterStrategy());
            convertOp.removeMultiClusterStrategyAttr();
        }
    };
    func.walk(callbackConvert);
}

}  // namespace

//
// createStrategyManagerImplPass
//

std::unique_ptr<mlir::Pass> createStrategyManagerImplPass(bool enablePrefetchTiling, Logger log) {
    return std::make_unique<StrategyManagerImplPass>(enablePrefetchTiling, log);
}

}  // namespace vpux::VPU
