//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/utils/VPU/tile_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"

#include <llvm/ADT/TypeSwitch.h>

namespace vpux {
namespace VPU {

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTileDistributions(
        VPU::NCEConvolutionOp origOp, SiblingOpsAnalysis& siblingsAnalysis, const TileInfo& outTile,
        const std::optional<InputTiling>& inputTiles) {
    const auto tiling = inputTiles.value_or(origOp.backInferTileInfo(outTile, Logger::global()));

    const auto tiles = tiling.tiles;
    VPUX_THROW_WHEN(tiles.size() < 2, "Not enough tiles {0} for operaion {1}", tiles.size(), origOp);

    auto inputTileType = origOp.getInput().getType().cast<vpux::NDTypeInterface>().extractDenseTile(tiles[0].offsets,
                                                                                                    tiles[0].shape);
    auto filterTileType = origOp.getFilter().getType().cast<vpux::NDTypeInterface>().extractDenseTile(tiles[1].offsets,
                                                                                                      tiles[1].shape);
    auto outputTileType =
            origOp.getType().cast<vpux::NDTypeInterface>().extractDenseTile(outTile.offsets, outTile.shape);

    if (origOp->hasAttr(VPU::multiClusterStrategy)) {
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                        origOp->getLoc());
        auto strategy = clusteredOp.getMultiClusterStrategy().value();

        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(nceOp == nullptr, "Op {0} has multiClusterStrategy but is not an NCEOp", origOp->getLoc());

        auto numClusters = VPU::getOptimalNumClusters(
                clusteredOp, outputTileType.getShape(),
                clusteredOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
        return {std::make_pair(inputTileType,
                               VPU::getActivationDistributionAttrFromOp(clusteredOp, inputTileType, numClusters,
                                                                        siblingsAnalysis, nullptr, tiles[0])),
                std::make_pair(filterTileType,
                               VPU::getFilterDistributionAttrFromOp(nceOp, filterTileType, numClusters, strategy)),
                std::make_pair(outputTileType,
                               VPU::getOutputDistributionAttrFromOp(clusteredOp, outputTileType, numClusters,
                                                                    siblingsAnalysis, {}, outTile))};
    }

    return {std::make_pair(inputTileType, TensorDistributionMap{}),
            std::make_pair(filterTileType, TensorDistributionMap{}),
            std::make_pair(outputTileType, TensorDistributionMap{})};
}

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTileDistributions(
        VPU::NCEMaxPoolOp origOp, SiblingOpsAnalysis& siblingsAnalysis, const TileInfo& outTile,
        const std::optional<InputTiling>& inputTiles) {
    const auto tiling = inputTiles.value_or(origOp.backInferTileInfo(outTile, Logger::global()));

    const auto tiles = tiling.tiles;
    VPUX_THROW_WHEN(tiles.empty(), "There are no tiles for operation {0}", origOp);

    auto inputTileType = origOp.getInput().getType().cast<vpux::NDTypeInterface>().extractDenseTile(tiles[0].offsets,
                                                                                                    tiles[0].shape);
    auto outputTileType =
            origOp.getType().cast<vpux::NDTypeInterface>().extractDenseTile(outTile.offsets, outTile.shape);

    if (origOp->hasAttr(VPU::multiClusterStrategy)) {
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                        origOp->getLoc());

        auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape(),
                                                      clusteredOp.getMultiClusterStrategy().value());
        return {std::make_pair(inputTileType,
                               VPU::getActivationDistributionAttrFromOp(clusteredOp, inputTileType, numClusters,
                                                                        siblingsAnalysis, nullptr, tiles[0])),
                std::make_pair(outputTileType,
                               VPU::getOutputDistributionAttrFromOp(clusteredOp, outputTileType, numClusters,
                                                                    siblingsAnalysis, {}, outTile))};
    }

    return {std::make_pair(inputTileType, TensorDistributionMap{}),
            std::make_pair(outputTileType, TensorDistributionMap{})};
}

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTileDistributions(
        VPU::NCEAveragePoolOp origOp, SiblingOpsAnalysis& siblingsAnalysis, const TileInfo& outTile,
        const std::optional<InputTiling>& inputTiles) {
    const auto tiling = inputTiles.value_or(origOp.backInferTileInfo(outTile, Logger::global()));

    const auto tiles = tiling.tiles;

    VPUX_THROW_WHEN(tiles.empty(), "There are no tiles for operaion {0}", origOp);

    auto inputTileType = origOp.getInput().getType().cast<vpux::NDTypeInterface>().extractDenseTile(tiles[0].offsets,
                                                                                                    tiles[0].shape);
    auto outputTileType =
            origOp.getType().cast<vpux::NDTypeInterface>().extractDenseTile(outTile.offsets, outTile.shape);

    if (origOp->hasAttr(VPU::multiClusterStrategy)) {
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                        origOp->getLoc());

        auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape(),
                                                      clusteredOp.getMultiClusterStrategy().value());
        return {std::make_pair(inputTileType,
                               VPU::getActivationDistributionAttrFromOp(clusteredOp, inputTileType, numClusters,
                                                                        siblingsAnalysis, nullptr, tiles[0])),
                std::make_pair(outputTileType,
                               VPU::getOutputDistributionAttrFromOp(clusteredOp, outputTileType, numClusters,
                                                                    siblingsAnalysis, {}, outTile))};
    }

    return {std::make_pair(inputTileType, TensorDistributionMap{}),
            std::make_pair(outputTileType, TensorDistributionMap{})};
}

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTileDistributions(
        VPU::NCEDepthConvolutionOp origOp, SiblingOpsAnalysis& siblingsAnalysis, const TileInfo& outTile,
        const std::optional<InputTiling>& inputTiles) {
    const auto tiling = inputTiles.value_or(origOp.backInferTileInfo(outTile, Logger::global()));

    const auto tiles = tiling.tiles;

    VPUX_THROW_WHEN(tiles.size() < 2, "There are not enough tiles {0} for operaion {1}", tiles.size(), origOp);
    auto inputTileType = origOp.getInput().getType().cast<vpux::NDTypeInterface>().extractDenseTile(tiles[0].offsets,
                                                                                                    tiles[0].shape);
    auto filterTileType = origOp.getFilter().getType().cast<vpux::NDTypeInterface>().extractDenseTile(tiles[1].offsets,
                                                                                                      tiles[1].shape);
    auto outputTileType =
            origOp.getType().cast<vpux::NDTypeInterface>().extractDenseTile(outTile.offsets, outTile.shape);

    if (origOp->hasAttr(VPU::multiClusterStrategy)) {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(nceOp == nullptr, "Op {0} has multiClusterStrategy but is not an NCEOp", origOp->getLoc());

        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                        origOp->getLoc());
        auto strategy = clusteredOp.getMultiClusterStrategy().value();
        auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape(), strategy);
        return {std::make_pair(inputTileType,
                               VPU::getActivationDistributionAttrFromOp(clusteredOp, inputTileType, numClusters,
                                                                        siblingsAnalysis, nullptr, tiles[0])),
                std::make_pair(filterTileType,
                               VPU::getFilterDistributionAttrFromOp(nceOp, filterTileType, numClusters, strategy)),
                std::make_pair(outputTileType,
                               VPU::getOutputDistributionAttrFromOp(clusteredOp, outputTileType, numClusters,
                                                                    siblingsAnalysis, {}, outTile))};
    }

    return {std::make_pair(inputTileType, TensorDistributionMap{}),
            std::make_pair(filterTileType, TensorDistributionMap{}),
            std::make_pair(outputTileType, TensorDistributionMap{})};
}

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTileDistributions(
        VPU::NCECompressConvolutionOp origOp, SiblingOpsAnalysis& siblingsAnalysis, const TileInfo& outTile,
        const std::optional<InputTiling>& inputTiles) {
    const auto tiles = inputTiles.value_or(origOp.backInferTileInfo(outTile, Logger::global())).tiles;
    auto inputTileType = origOp.getInput().getType().cast<vpux::NDTypeInterface>().extractDenseTile(tiles[0].offsets,
                                                                                                    tiles[0].shape);
    auto filterTileType = origOp.getFilter().getType().cast<vpux::NDTypeInterface>().extractDenseTile(tiles[1].offsets,
                                                                                                      tiles[1].shape);
    auto outputTileType =
            origOp.getType().cast<vpux::NDTypeInterface>().extractDenseTile(outTile.offsets, outTile.shape);

    if (origOp->hasAttr(VPU::multiClusterStrategy)) {
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                        origOp->getLoc());
        auto strategy = clusteredOp.getMultiClusterStrategy().value();

        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(nceOp == nullptr, "Op {0} has multiClusterStrategy but is not an NCEOp", origOp->getLoc());

        auto numClusters = VPU::getOptimalNumClusters(
                clusteredOp, outputTileType.getShape(),
                clusteredOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
        return {std::make_pair(inputTileType,
                               VPU::getActivationDistributionAttrFromOp(clusteredOp, inputTileType, numClusters,
                                                                        siblingsAnalysis, nullptr, tiles[0])),
                std::make_pair(filterTileType,
                               VPU::getFilterDistributionAttrFromOp(nceOp, filterTileType, numClusters, strategy)),
                std::make_pair(outputTileType,
                               VPU::getOutputDistributionAttrFromOp(clusteredOp, outputTileType, numClusters,
                                                                    siblingsAnalysis, {}, outTile))};
    }

    return {std::make_pair(inputTileType, TensorDistributionMap{}),
            std::make_pair(filterTileType, TensorDistributionMap{}),
            std::make_pair(outputTileType, TensorDistributionMap{})};
}

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTileDistributions(
        VPU::NCEPermuteOp origOp, SiblingOpsAnalysis& siblingsAnalysis, const TileInfo& outTile,
        const std::optional<InputTiling>& inputTiles) {
    const auto tiles = inputTiles.value_or(origOp.backInferTileInfo(outTile, Logger::global())).tiles;
    auto inputTileType = origOp.getInput().getType().cast<vpux::NDTypeInterface>().extractDenseTile(tiles[0].offsets,
                                                                                                    tiles[0].shape);
    auto outputTileType =
            origOp.getType().cast<vpux::NDTypeInterface>().extractDenseTile(outTile.offsets, outTile.shape);

    if (origOp->hasAttr(VPU::multiClusterStrategy)) {
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                        origOp->getLoc());

        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(nceOp == nullptr, "Op {0} has multiClusterStrategy but is not an NCEOp", origOp->getLoc());

        auto numClusters = VPU::getOptimalNumClusters(
                clusteredOp, outputTileType.getShape(),
                clusteredOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
        return {std::make_pair(inputTileType,
                               VPU::getActivationDistributionAttrFromOp(clusteredOp, inputTileType, numClusters,
                                                                        siblingsAnalysis, nullptr, tiles[0])),
                std::make_pair(outputTileType,
                               VPU::getOutputDistributionAttrFromOp(clusteredOp, outputTileType, numClusters,
                                                                    siblingsAnalysis, {}, outTile))};
    }

    return {std::make_pair(inputTileType, TensorDistributionMap{}),
            std::make_pair(outputTileType, TensorDistributionMap{})};
}

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTileDistributions(
        VPU::NCEInterpolateOp origOp, SiblingOpsAnalysis& siblingsAnalysis, const TileInfo& outTile,
        const std::optional<InputTiling>& inputTiles) {
    const auto tiling = inputTiles.value_or(origOp.backInferTileInfo(outTile, Logger::global()));

    const auto tiles = tiling.tiles;
    VPUX_THROW_WHEN(tiles.size() < 2, "Not enough tiles {0} for operaion {1}", tiles.size(), origOp);

    auto inputTileType = origOp.getInput().getType().cast<vpux::NDTypeInterface>().extractDenseTile(tiles[0].offsets,
                                                                                                    tiles[0].shape);
    auto filterTileType = origOp.getWeights().getType().cast<vpux::NDTypeInterface>().extractDenseTile(tiles[1].offsets,
                                                                                                       tiles[1].shape);
    auto outputTileType =
            origOp.getType().cast<vpux::NDTypeInterface>().extractDenseTile(outTile.offsets, outTile.shape);

    if (origOp->hasAttr(VPU::multiClusterStrategy)) {
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                        origOp->getLoc());
        auto strategy = clusteredOp.getMultiClusterStrategy().value();

        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(nceOp == nullptr, "Op {0} has multiClusterStrategy but is not an NCEOp", origOp->getLoc());

        auto numClusters = VPU::getOptimalNumClusters(
                clusteredOp, outputTileType.getShape(),
                clusteredOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
        return {std::make_pair(inputTileType,
                               VPU::getActivationDistributionAttrFromOp(clusteredOp, inputTileType, numClusters,
                                                                        siblingsAnalysis, nullptr, tiles[0])),
                std::make_pair(filterTileType,
                               VPU::getFilterDistributionAttrFromOp(nceOp, filterTileType, numClusters, strategy)),
                std::make_pair(outputTileType,
                               VPU::getOutputDistributionAttrFromOp(clusteredOp, outputTileType, numClusters,
                                                                    siblingsAnalysis, {}, outTile))};
    }

    return {std::make_pair(inputTileType, TensorDistributionMap{}),
            std::make_pair(filterTileType, TensorDistributionMap{}),
            std::make_pair(outputTileType, TensorDistributionMap{})};
}

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTileDistributionsCommon(
        mlir::Operation* origOp, SiblingOpsAnalysis& siblingsAnalysis, const TileInfo& outTile,
        const std::optional<InputTiling>& inputTiles) {
    const auto outputType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();

    SmallVector<vpux::TileInfo> inTiles{outTile};
    if (!inputTiles.has_value()) {
        if (auto tilingBuilderInterface = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(origOp)) {
            inTiles = tilingBuilderInterface.backInferTileInfo(outTile, Logger::global()).tiles;
        }
    } else if (!inputTiles.value().tiles.empty()) {
        inTiles = inputTiles.value().tiles;
    }

    VPUX_THROW_UNLESS(inTiles.size() == origOp->getOperands().size(),
                      "Unexpected SW inputTile size '{0}' and Op operands size '{1}'", inTiles.size(),
                      origOp->getOperands().size());

    std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> inputTileTypes;
    for (const auto& input : origOp->getOperands() | indexed) {
        const auto inputType = input.value().getType().cast<vpux::NDTypeInterface>();
        auto inputTileType = inputType.extractDenseTile(inTiles[input.index()].offsets, inTiles[input.index()].shape);
        inputTileTypes.push_back(std::make_pair(inputTileType, TensorDistributionMap{}));
    }
    const auto outputTileType = outputType.extractDenseTile(outTile.offsets, outTile.shape);

    if (!origOp->hasAttr(VPU::multiClusterStrategy)) {
        inputTileTypes.push_back(std::make_pair(outputTileType, TensorDistributionMap{}));
        return inputTileTypes;
    }

    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp);
    VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                    origOp->getLoc());
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape(),
                                                  clusteredOp.getMultiClusterStrategy().value());

    std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> distributedTensorTypes;
    SmallVector<NDTypeInterface> inputTypes;
    for (const auto& inputTileType : inputTileTypes) {
        auto inDistribution = VPU::getActivationDistributionAttrFromOp(clusteredOp, inputTileType.first, numClusters,
                                                                       siblingsAnalysis, outputTileType);
        distributedTensorTypes.push_back(std::make_pair(inputTileType.first, inDistribution));
        inputTypes.push_back(inputTileType.first);
    }

    auto outDistribution = VPU::getOutputDistributionAttrFromOp(clusteredOp, outputTileType, numClusters,
                                                                siblingsAnalysis, inputTypes);
    distributedTensorTypes.push_back(std::make_pair(outputTileType, outDistribution));

    return distributedTensorTypes;
}

std::vector<std::pair<NDTypeInterface, TensorDistributionMap>> getTileDistributions(
        mlir::Operation* op, SiblingOpsAnalysis& siblingsAnalysis, const TileInfo& outTile,
        const std::optional<InputTiling>& inputTiles) {
    if (auto convOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(op)) {
        return getTileDistributions(convOp, siblingsAnalysis, outTile, inputTiles);
    }
    if (auto convOp = mlir::dyn_cast<VPU::NCECompressConvolutionOp>(op)) {
        return getTileDistributions(convOp, siblingsAnalysis, outTile, inputTiles);
    }
    if (auto poolOp = mlir::dyn_cast<VPU::NCEMaxPoolOp>(op)) {
        return getTileDistributions(poolOp, siblingsAnalysis, outTile, inputTiles);
    }
    if (auto poolOp = mlir::dyn_cast<VPU::NCEAveragePoolOp>(op)) {
        return getTileDistributions(poolOp, siblingsAnalysis, outTile, inputTiles);
    }
    if (auto depthConvOp = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(op)) {
        return getTileDistributions(depthConvOp, siblingsAnalysis, outTile, inputTiles);
    }
    if (auto interpOp = mlir::dyn_cast<VPU::NCEInterpolateOp>(op)) {
        return getTileDistributions(interpOp, siblingsAnalysis, outTile, inputTiles);
    }
    if (auto permuteOp = mlir::dyn_cast<VPU::NCEPermuteOp>(op)) {
        return getTileDistributions(permuteOp, siblingsAnalysis, outTile, inputTiles);
    }

    auto tileConf = inputTiles.value_or(vpux::backInferEltwiseTile(op, outTile));

    return getTileDistributionsCommon(op, siblingsAnalysis, outTile, tileConf);
}

// Convolution

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::ConvolutionOp origOp, const TileInfo& outTile,
                                                const std::optional<InputTiling>& inputTiles) {
    const auto origBiasShape = origOp.getBias() != nullptr ? getShape(origOp.getBias()) : ShapeRef();
    const auto origPadding = PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd());

    auto tileConf = inputTiles.value_or(vpux::backInferConvTile(outTile, getShape(origOp.getInput()),
                                                                getShape(origOp.getFilter()), origBiasShape,
                                                                origOp.getStrides(), origPadding));

    SmallVector<vpux::NDTypeInterface> tileTypes;

    tileTypes.push_back(origOp.getInput().getType().cast<vpux::NDTypeInterface>().extractDenseTile(
            tileConf.tiles[0].offsets, tileConf.tiles[0].shape));
    tileTypes.push_back(origOp.getFilter().getType().cast<vpux::NDTypeInterface>().extractDenseTile(
            tileConf.tiles[1].offsets, tileConf.tiles[1].shape));
    tileTypes.push_back(
            origOp.getType().cast<vpux::NDTypeInterface>().extractDenseTile(outTile.offsets, outTile.shape));

    return tileTypes;
}

// MaxPool

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::MaxPoolOp origOp, const TileInfo& outTile,
                                                const std::optional<InputTiling>& inputTiles) {
    const auto origPadding = PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd());

    auto tileConf = inputTiles.value_or(vpux::backInferPoolTile(
            outTile, getShape(origOp.getInput()), origOp.getKernelSize(), origOp.getStrides(), origPadding));

    SmallVector<vpux::NDTypeInterface> tileTypes;
    VPUX_THROW_WHEN(tileConf.tiles.empty(), "There are no tiles for operaion {0}", origOp);

    tileTypes.push_back(origOp.getInput().getType().cast<vpux::NDTypeInterface>().extractDenseTile(
            tileConf.tiles[0].offsets, tileConf.tiles[0].shape));
    tileTypes.push_back(
            origOp.getType().cast<vpux::NDTypeInterface>().extractDenseTile(outTile.offsets, outTile.shape));

    return tileTypes;
}

// GroupConvolution

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::GroupConvolutionOp origOp, const TileInfo& outTile,
                                                const std::optional<InputTiling>& inputTiles) {
    const auto origBiasShape = origOp.getBias() != nullptr ? getShape(origOp.getBias()) : ShapeRef();
    const auto origPadding = PadInfo(origOp.getPadsBegin(), origOp.getPadsEnd());
    const auto origGroups = origOp.getGroups().value_or(1);

    auto tileConf = inputTiles.value_or(vpux::backInferGroupConvTile(outTile, getShape(origOp.getInput()),
                                                                     getShape(origOp.getFilter()), origBiasShape,
                                                                     origOp.getStrides(), origPadding, origGroups));

    VPUX_THROW_WHEN(tileConf.tiles.size() < 2, "There are not enough tiles {0} for operaion {1}", tileConf.tiles.size(),
                    origOp);
    SmallVector<vpux::NDTypeInterface> tileTypes;

    tileTypes.push_back(origOp.getInput().getType().cast<vpux::NDTypeInterface>().extractDenseTile(
            tileConf.tiles[0].offsets, tileConf.tiles[0].shape));
    tileTypes.push_back(origOp.getFilter().getType().cast<vpux::NDTypeInterface>().extractDenseTile(
            tileConf.tiles[1].offsets, tileConf.tiles[1].shape));
    tileTypes.push_back(
            origOp.getType().cast<vpux::NDTypeInterface>().extractDenseTile(outTile.offsets, outTile.shape));

    return tileTypes;
}

// DepthToSpace

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::DepthToSpaceOp origOp, const TileInfo& outTile,
                                                const std::optional<InputTiling>& inputTiles) {
    const auto tiling = inputTiles.value_or(origOp.backInferTileInfo(outTile, Logger::global()));

    const auto tiles = tiling.tiles;

    VPUX_THROW_WHEN(tiles.empty(), "There are no tiles for operaion {0}", origOp->getLoc());

    auto inputTileType = origOp.getInput().getType().cast<vpux::NDTypeInterface>().extractDenseTile(tiles[0].offsets,
                                                                                                    tiles[0].shape);
    auto outputTileType =
            origOp.getType().cast<vpux::NDTypeInterface>().extractDenseTile(outTile.offsets, outTile.shape);

    if (origOp->hasAttr(VPU::multiClusterStrategy)) {
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                        origOp->getLoc());

        auto numClusters = VPU::getOptimalNumClusters(
                clusteredOp, outputTileType.getShape(),
                clusteredOp->getAttr(VPU::multiClusterStrategy).cast<VPU::MultiClusterStrategyAttr>().getValue());
        return {VPU::getDistributedActivationTypeFromOp(clusteredOp, inputTileType, numClusters, nullptr, tiles[0]),
                VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters, {}, outTile)};
    }

    return {inputTileType, outputTileType};
}

SmallVector<vpux::NDTypeInterface> getTileTypesCommon(mlir::Operation* origOp, const TileInfo& outTile,
                                                      const std::optional<InputTiling>& inputTiles) {
    const auto outputType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();

    SmallVector<vpux::TileInfo> inTiles{outTile};
    if (!inputTiles.has_value()) {
        if (auto tilingBuilderInterface = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(origOp)) {
            inTiles = tilingBuilderInterface.backInferTileInfo(outTile, Logger::global()).tiles;
        }
    } else if (!inputTiles.value().tiles.empty()) {
        inTiles = inputTiles.value().tiles;
    }

    VPUX_THROW_UNLESS(inTiles.size() == origOp->getOperands().size(),
                      "Unexpected SW inputTile size '{0}' and Op operands size '{1}'", inTiles.size(),
                      origOp->getOperands().size());

    mlir::SmallVector<vpux::NDTypeInterface> inputTileTypes;
    for (const auto& input : origOp->getOperands() | indexed) {
        const auto inputType = input.value().getType().cast<vpux::NDTypeInterface>();
        inputTileTypes.push_back(
                inputType.extractDenseTile(inTiles[input.index()].offsets, inTiles[input.index()].shape));
    }
    const auto outputTileType = outputType.extractDenseTile(outTile.offsets, outTile.shape);

    if (!origOp->hasAttr(VPU::multiClusterStrategy)) {
        inputTileTypes.push_back(outputTileType);
        return inputTileTypes;
    }

    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp);
    VPUX_THROW_WHEN(clusteredOp == nullptr, "Op {0} has multiClusterStrategy but is not an ClusteredOp",
                    origOp->getLoc());
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTileType.getShape(),
                                                  clusteredOp.getMultiClusterStrategy().value());

    SmallVector<vpux::NDTypeInterface> distributedTensorTypes;
    for (const auto& inputTileType : inputTileTypes) {
        auto inDistributedType =
                VPU::getDistributedActivationTypeFromOp(clusteredOp, inputTileType, numClusters, outputTileType);
        distributedTensorTypes.push_back(inDistributedType.cast<vpux::NDTypeInterface>());
    }

    auto outDistributedType =
            VPU::getDistributedOutputTypeFromOp(clusteredOp, outputTileType, numClusters, inputTileTypes);
    distributedTensorTypes.push_back(outDistributedType.cast<vpux::NDTypeInterface>());

    return distributedTensorTypes;
}

SmallVector<vpux::NDTypeInterface> getTileTypes(VPU::SWOpInterface origOp, const TileInfo& outTile,
                                                const std::optional<InputTiling>& inputTiles) {
    VPUX_THROW_UNLESS(origOp->getResults().size() == 1, "Only support SW with one output, but got '{0}'",
                      origOp->getResults().size());

    return getTileTypesCommon(origOp, outTile, inputTiles);
}

SmallVector<vpux::NDTypeInterface> getTileTypes(mlir::Operation* op, const TileInfo& outTile,
                                                const std::optional<InputTiling>& inputTiles) {
    if (auto convOp = mlir::dyn_cast<VPU::ConvolutionOp>(op)) {
        return getTileTypes(convOp, outTile, inputTiles);
    }
    if (auto convOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(op)) {
        return getTileTypes(convOp, outTile, inputTiles);
    }
    if (auto convOp = mlir::dyn_cast<VPU::NCECompressConvolutionOp>(op)) {
        return getTileTypes(convOp, outTile, inputTiles);
    }
    if (auto poolOp = mlir::dyn_cast<VPU::MaxPoolOp>(op)) {
        return getTileTypes(poolOp, outTile, inputTiles);
    }
    if (auto poolOp = mlir::dyn_cast<VPU::NCEMaxPoolOp>(op)) {
        return getTileTypes(poolOp, outTile, inputTiles);
    }
    if (auto poolOp = mlir::dyn_cast<VPU::NCEAveragePoolOp>(op)) {
        return getTileTypes(poolOp, outTile, inputTiles);
    }
    if (auto groupConvOp = mlir::dyn_cast<VPU::GroupConvolutionOp>(op)) {
        return getTileTypes(groupConvOp, outTile, inputTiles);
    }
    if (auto depthConvOp = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(op)) {
        return getTileTypes(depthConvOp, outTile, inputTiles);
    }
    if (auto depthToSpaceOp = mlir::dyn_cast<VPU::DepthToSpaceOp>(op)) {
        return getTileTypes(depthToSpaceOp, outTile, inputTiles);
    }
    if (auto swOp = mlir::dyn_cast<VPU::SWOpInterface>(op)) {
        return getTileTypes(swOp, outTile, inputTiles);
    }
    if (auto interpOp = mlir::dyn_cast<VPU::NCEInterpolateOp>(op)) {
        return getTileTypes(interpOp, outTile, inputTiles);
    }
    if (auto permuteOp = mlir::dyn_cast<VPU::NCEPermuteOp>(op)) {
        return getTileTypes(permuteOp, outTile, inputTiles);
    }
    if (auto gatherOp = mlir::dyn_cast<VPU::GatherOp>(op)) {
        return getTileTypesCommon(gatherOp, outTile, inputTiles);
    }

    auto tileConf = inputTiles.value_or(vpux::backInferEltwiseTile(op, outTile));

    return getTileTypesCommon(op, outTile, tileConf);
}

Byte getRequiredCMXForWeight(VPU::ConvolutionOp convOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(convOp, tiling, inputTiles);
    const auto lastFilterTileType = tileTypes[1];
    return getRequiredCMXSize({lastFilterTileType});
}

Byte getRequiredCMXForWeight(VPU::NCEConvolutionOp convOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(convOp, tiling, inputTiles);
    const auto lastFilterTileType = tileTypes[1];
    const auto outputTileType = tileTypes[2];
    const auto OC = outputTileType.getShape()[Dims4D::Act::C];
    return getRequiredCMXSizeForNCEOps({lastFilterTileType}, OC);
}

Byte getRequiredCMXForWeight(VPU::NCECompressConvolutionOp convOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(convOp.getOperation(), tiling, inputTiles);
    const auto lastFilterTileType = tileTypes[1];
    const auto outputTileType = tileTypes[2];
    const auto OC = outputTileType.getShape()[Dims4D::Act::C];
    return getRequiredCMXSizeForNCEOps({lastFilterTileType}, OC);
}

Byte getRequiredCMX(VPU::ConvolutionOp convOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(convOp, tiling, inputTiles);
    const auto lastInputTileType = tileTypes[0];
    const auto lastFilterTileType = tileTypes[1];
    const auto lastOutputTileType = tileTypes[2];
    return getRequiredCMXSize({lastInputTileType, lastFilterTileType, lastOutputTileType});
}

Byte getRequiredCMX(VPU::NCEConvolutionOp convOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(convOp, tiling, inputTiles);
    const auto lastInputTileType = tileTypes[0];
    const auto lastFilterTileType = tileTypes[1];
    const auto lastOutputTileType = tileTypes[2];
    const auto OC = lastOutputTileType.getShape()[Dims4D::Act::C];
    return getRequiredCMXSizeForNCEOps({lastInputTileType, lastFilterTileType, lastOutputTileType}, OC);
}

Byte getRequiredCMX(VPU::NCECompressConvolutionOp convOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(convOp.getOperation(), tiling, inputTiles);
    const auto lastInputTileType = tileTypes[0];
    const auto lastFilterTileType = tileTypes[1];
    const auto lastOutputTileType = tileTypes[2];
    const auto OC = lastOutputTileType.getShape()[Dims4D::Act::C];
    return getRequiredCMXSizeForNCEOps({lastInputTileType, lastFilterTileType, lastOutputTileType}, OC);
}

Byte getRequiredCMXForWeight(VPU::GroupConvolutionOp gConvOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(gConvOp, tiling, inputTiles);
    const auto filterTileType = tileTypes[1];
    return getRequiredCMXSize({filterTileType});
}

Byte getRequiredCMXForWeight(VPU::NCEDepthConvolutionOp gConvOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(gConvOp, tiling, inputTiles);
    const auto filterTileShape = tileTypes[1];
    const auto outputTileType = tileTypes[2];
    const auto OC = outputTileType.getShape()[Dims4D::Act::C];
    return getRequiredCMXSizeForNCEOps({filterTileShape}, OC);
}

Byte getRequiredCMX(VPU::GroupConvolutionOp gConvOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(gConvOp, tiling, inputTiles);
    const auto inputTileType = tileTypes[0];
    return getRequiredCMXSize({inputTileType, inputTileType}) + getRequiredCMXForWeight(gConvOp, tiling, inputTiles);
}

Byte getRequiredCMX(VPU::NCEDepthConvolutionOp dConvOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(dConvOp, tiling, inputTiles);
    const auto inputTileType = tileTypes[0];
    const auto filterShape = Shape(parseIntArrayAttr<int64_t>(dConvOp.getRawFilterShape()));
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];
    const Shape kernelSizeVals{KY, KX};
    auto kernelStrides = dConvOp.getStrides();
    const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(kernelStrides));

    return getRequiredCMXSizeForNCEOps({inputTileType, inputTileType}, 0) +
           getRequiredCMXForWeight(dConvOp, tiling, inputTiles);
}

Byte getRequiredCMX(VPU::SWOpInterface swOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(swOp, tiling, inputTiles);
    return getRequiredCMXSize(tileTypes);
}

Byte getRequiredCMX(VPU::DepthToSpaceOp d2sOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(d2sOp, tiling, inputTiles);
    return getRequiredCMXSize(tileTypes);
}

Byte getRequiredCMXForWeight(VPU::MaxPoolOp /*op*/, const vpux::TileInfo& /*tiling*/,
                             const std::optional<InputTiling>& /*inputTiles*/) {
    return Byte(0);
}

Byte getRequiredCMXForWeight(VPU::NCEPermuteOp /*op*/, const vpux::TileInfo& /*tiling*/,
                             const std::optional<InputTiling>& /*inputTiles*/) {
    return Byte(0);
}

Byte getRequiredCMXForWeight(VPU::NCEMaxPoolOp /*op*/, const vpux::TileInfo& /*tiling*/,
                             const std::optional<InputTiling>& /*inputTiles*/) {
    return Byte(0);
}

Byte getRequiredCMXForWeight(VPU::NCEAveragePoolOp /*op*/, const vpux::TileInfo& /*tiling*/,
                             const std::optional<InputTiling>& /*inputTiles*/) {
    return Byte(0);
}

Byte getRequiredCMX(VPU::MaxPoolOp poolOp, const vpux::TileInfo& tiling, const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(poolOp.getOperation(), tiling, inputTiles);
    auto inputType = tileTypes[0];
    auto outputType = tileTypes[1];
    return getRequiredCMXSize({inputType, outputType});
}

Byte getRequiredCMX(VPU::NCEMaxPoolOp poolOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(poolOp.getOperation(), tiling, inputTiles);
    auto inputType = tileTypes[0];
    auto outputType = tileTypes[1];
    auto kernelSize = poolOp.getKernelSize();
    auto kernelStrides = poolOp.getStrides();
    const auto inputShape = inputType.getShape();
    const auto IC = inputShape[Dims4D::Act::C];

    const auto kernelSizeVals = Shape(parseIntArrayAttr<int64_t>(kernelSize));
    const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(kernelStrides));

    return getRequiredCMXSizeForNCEOps({inputType, outputType}, IC);
}

Byte getRequiredCMX(VPU::NCEPermuteOp pqOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(pqOp, tiling, inputTiles);
    auto inputType = tileTypes[0];
    auto outputType = tileTypes[1];
    return getRequiredCMXSize({inputType, outputType});
}

Byte getRequiredCMX(VPU::NCEAveragePoolOp poolOp, const vpux::TileInfo& tiling,
                    const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(poolOp.getOperation(), tiling, inputTiles);
    auto inputType = tileTypes[0];
    auto outputType = tileTypes[1];
    auto kernelSize = poolOp.getKernelSize();
    auto kernelStrides = poolOp.getStrides();
    const auto inputShape = inputType.getShape();
    const auto IC = inputShape[Dims4D::Act::C];

    const auto kernelSizeVals = Shape(parseIntArrayAttr<int64_t>(kernelSize));
    const auto kernelStridesVals = Shape(parseIntArrayAttr<int64_t>(kernelStrides));

    return getRequiredCMXSizeForNCEOps({inputType, outputType}, IC);
}

Byte getEltwiseRequiredCMX(mlir::Operation* op, const vpux::TileInfo& tiling,
                           const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(op, tiling, inputTiles);
    auto firstInputType = tileTypes[0];
    auto secondInputType = tileTypes[1];
    auto outputType = tileTypes[2];
    // Inplace eltwise requires less CMX
    if (auto nceEltwise = mlir::dyn_cast<VPU::NCEEltwiseOp>(op)) {
        if (nceEltwise.getIsInplace().value_or(false)) {
            return getRequiredCMXSize({firstInputType, secondInputType});
        }
    }
    // Two inputs are the same, require less CMX
    if (op->getOperand(0) == op->getOperand(1)) {
        VPUX_THROW_WHEN(firstInputType != secondInputType, "Input tile is different for eltwise input");
        return getRequiredCMXSize({firstInputType, outputType});
    }

    return getRequiredCMXSize({firstInputType, secondInputType, outputType});
}

Byte getRequiredCMX(VPU::AddOp op, const vpux::TileInfo& tiling, const std::optional<InputTiling>& inputTiles) {
    return getEltwiseRequiredCMX(op.getOperation(), tiling, inputTiles);
}

Byte getRequiredCMXForWeight(VPU::AddOp /*op*/, const vpux::TileInfo& /*tiling*/,
                             const std::optional<InputTiling>& /*inputTiles*/) {
    return Byte(0);
}

Byte getRequiredCMX(VPU::MultiplyOp op, const vpux::TileInfo& tiling, const std::optional<InputTiling>& inputTiles) {
    return getEltwiseRequiredCMX(op.getOperation(), tiling, inputTiles);
}

Byte getRequiredCMXForWeight(VPU::MultiplyOp /*op*/, const vpux::TileInfo& /*tiling*/,
                             const std::optional<InputTiling>& /*inputTiles*/) {
    return Byte(0);
}

Byte getRequiredCMX(VPU::SubtractOp op, const vpux::TileInfo& tiling, const std::optional<InputTiling>& inputTiles) {
    return getEltwiseRequiredCMX(op.getOperation(), tiling, inputTiles);
}

Byte getRequiredCMXForWeight(VPU::SubtractOp /*op*/, const vpux::TileInfo& /*tiling*/,
                             const std::optional<InputTiling>& /*inputTiles*/) {
    return Byte(0);
}

Byte getRequiredCMX(VPU::AndOp op, const vpux::TileInfo& tiling, const std::optional<InputTiling>& inputTiles) {
    return getEltwiseRequiredCMX(op.getOperation(), tiling, inputTiles);
}

Byte getRequiredCMXForWeight(VPU::AndOp /*op*/, const vpux::TileInfo& /*tiling*/,
                             const std::optional<InputTiling>& /*inputTiles*/) {
    return Byte(0);
}

Byte getRequiredCMX(VPU::NCEEltwiseOp op, const vpux::TileInfo& tiling, const std::optional<InputTiling>& inputTiles) {
    return getEltwiseRequiredCMX(op.getOperation(), tiling, inputTiles);
}

Byte getRequiredCMXForWeight(VPU::NCEEltwiseOp /*op*/, const vpux::TileInfo& /*tiling*/,
                             const std::optional<InputTiling>& /*inputTiles*/) {
    return Byte(0);
}

Byte getRequiredCMXForWeight(VPU::NCEInterpolateOp NCEInterpOp, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles) {
    auto tileTypes = getTileTypes(NCEInterpOp, tiling, inputTiles);
    const auto filterTileShape = tileTypes[1];
    const auto outputTileType = tileTypes[2];
    const auto OC = outputTileType.getShape()[Dims4D::Act::C];
    return getRequiredCMXSizeForNCEOps({filterTileShape}, OC);
}

Byte getRequiredCMXForWeight(mlir::Operation* op, const vpux::TileInfo& tiling,
                             const std::optional<InputTiling>& inputTiles) {
    return llvm::TypeSwitch<mlir::Operation*, Byte>(op)
            .Case<VPU::ConvolutionOp>([&](VPU::ConvolutionOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCECompressConvolutionOp>([&](VPU::NCECompressConvolutionOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::MaxPoolOp>([&](VPU::MaxPoolOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEAveragePoolOp>([&](VPU::NCEAveragePoolOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::AddOp>([&](VPU::AddOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::MultiplyOp>([&](VPU::MultiplyOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::SubtractOp>([&](VPU::SubtractOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::AndOp>([&](VPU::AndOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::GroupConvolutionOp>([&](VPU::GroupConvolutionOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEInterpolateOp>([&](VPU::NCEInterpolateOp origOp) {
                return getRequiredCMXForWeight(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEPermuteOp>([&](VPU::NCEPermuteOp pqOp) {
                return getRequiredCMXForWeight(pqOp, tiling, inputTiles);
            })
            .Default([](mlir::Operation* unknownOp) -> Byte {
                VPUX_THROW("Operation CMX check '{0}' at '{1}' is not implemented", unknownOp->getName(),
                           unknownOp->getLoc());
            });
}

Byte getRequiredCMX(mlir::Operation* op, const vpux::TileInfo& tiling, Logger log,
                    const std::optional<InputTiling>& inputTiles) {
    return llvm::TypeSwitch<mlir::Operation*, Byte>(op)
            .Case<VPU::ConvolutionOp>([&](VPU::ConvolutionOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEConvolutionOp>([&](VPU::NCEConvolutionOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCECompressConvolutionOp>([&](VPU::NCECompressConvolutionOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::MaxPoolOp>([&](VPU::MaxPoolOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEMaxPoolOp>([&](VPU::NCEMaxPoolOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEAveragePoolOp>([&](VPU::NCEAveragePoolOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::AddOp>([&](VPU::AddOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::MultiplyOp>([&](VPU::MultiplyOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::SubtractOp>([&](VPU::SubtractOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::AndOp>([&](VPU::AndOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::GroupConvolutionOp>([&](VPU::GroupConvolutionOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEDepthConvolutionOp>([&](VPU::NCEDepthConvolutionOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::SWOpInterface>([&](VPU::SWOpInterface origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::DepthToSpaceOp>([&](VPU::DepthToSpaceOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Case<VPU::NCEPermuteOp>([&](VPU::NCEPermuteOp origOp) {
                return getRequiredCMX(origOp, tiling, inputTiles);
            })
            .Default([&](mlir::Operation* defaultOp) -> Byte {
                log.trace("getRequiredCMX is not implemented for op {0}, use default function and ignore parent tiling",
                          defaultOp->getName());
                return getRequiredCMXSizeForDefaultOps(defaultOp);
            });
}

Byte getRequiredCMXSize(ArrayRef<vpux::NDTypeInterface> operands) {
    Byte requiredCMX(0);

    for (const auto& operand : operands) {
        requiredCMX += operand.getTotalAllocSize();
    }

    return requiredCMX;
}

Byte getRequiredCMXSizeForNCEOps(ArrayRef<vpux::NDTypeInterface> operands, int64_t numChannels) {
    auto requiredCMX = getRequiredCMXSize(operands);

    requiredCMX += numChannels * VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC * 4_Byte;

    return requiredCMX;
}

Byte getRequiredCMXSizeForDefaultOps(mlir::Operation* op) {
    SmallVector<vpux::NDTypeInterface> operands;
    auto getTypeFromValue = [](mlir::Value operand) {
        return operand.getType().cast<vpux::NDTypeInterface>();
    };
    std::transform(op->getOperands().begin(), op->getOperands().end(), std::back_inserter(operands), getTypeFromValue);
    std::transform(op->getResults().begin(), op->getResults().end(), std::back_inserter(operands), getTypeFromValue);
    auto requiredCMX = getRequiredCMXSize(operands);

    return requiredCMX;
}

OutputTiling getUniqueShapeTilingCandidates(mlir::Operation* op, const OutputTiling& origTiles, Logger) {
    if (origTiles.size() <= 2) {
        return origTiles;
    }

    return llvm::TypeSwitch<mlir::Operation*, OutputTiling>(op)
            .Case<VPU::NCEConvolutionOp, VPU::NCEDepthConvolutionOp, VPU::NCECompressConvolutionOp,
                  VPU::NCEAveragePoolOp, VPU::NCEMaxPoolOp, VPU::NCEPermuteOp>([&](mlir::Operation*) {
                auto axis = origTiles[0].axis;
                auto tilingDims = getNonOneDim(axis);
                const auto isTiledOnSpatialDims = [](auto dim) {
                    return dim.ind() >= Dims4D::Act::getSpatialDim(0).ind();
                };
                auto tileOnSpatialDimCnts = llvm::count_if(tilingDims, isTiledOnSpatialDims);
                // Return all tiles for checking if layer is tiled on both H and W
                if (tileOnSpatialDimCnts > 1) {
                    return origTiles;
                }

                // For NCE operations we need to account for the fact that padding is not occupying
                // any space in CMX so output tiles that are located on the border might result in
                // a smaller input tile than for an output tile located in the middle of the space
                // even if output tile shape is the same. To cover those cases add tiles with the same
                // shape located at the beggining of the dimension(to cover pad before), middle, and
                // end of dimension(to cover pad after)
                // Example:
                // origTiles:
                // shape = [1, 256, 65, 65] offset = [0, 0, 0, 0]
                // [1, 256, 65, 65] [0, 0, 65, 0]
                // [1, 256, 65, 65] [0, 0, 130, 0]
                // [1, 256, 65, 65] [0, 0, 195, 0]
                // [1, 256, 65, 65] [0, 0, 260, 0]
                //
                // gets transformed to uniqueShapeTiles:
                //  shape = [1, 256, 65, 65] offset = [0, 0, 0, 0] - first tile
                // [1, 256, 65, 65] [0, 0, 65, 0] - middle tile
                // [1, 256, 65, 65] [0, 0, 260, 0] - last tile

                auto lastTile = origTiles.back();
                const auto lastYOffset = lastTile.offsets[Dims4D::Act::getSpatialDim(0)];
                const auto lastXOffset = lastTile.offsets[Dims4D::Act::getSpatialDim(1)];

                enum TilePosition { First = 0, Middle = 1, Last = 2 };

                auto classifyTilePosition = [](int tileOffset, int lastOffset) {
                    if (tileOffset == 0) {
                        return TilePosition::First;
                    } else if (tileOffset != lastOffset) {
                        return TilePosition::Middle;
                    } else {
                        return TilePosition::Last;
                    }
                };

                auto isNCETileUnique = [&](const TileInfo& tile1, const TileInfo& tile2) {
                    if (tile1.shape == tile2.shape) {
                        auto tile1YOffset = tile1.offsets[Dims4D::Act::getSpatialDim(0)];
                        auto tile2YOffset = tile2.offsets[Dims4D::Act::getSpatialDim(0)];
                        auto tile1XOffset = tile1.offsets[Dims4D::Act::getSpatialDim(1)];
                        auto tile2XOffset = tile2.offsets[Dims4D::Act::getSpatialDim(1)];
                        if (tile1YOffset != tile2YOffset) {
                            return classifyTilePosition(tile1YOffset, lastYOffset) <
                                   classifyTilePosition(tile2YOffset, lastYOffset);
                        }
                        if (tile1XOffset != tile2XOffset) {
                            return classifyTilePosition(tile1XOffset, lastXOffset) <
                                   classifyTilePosition(tile2XOffset, lastXOffset);
                        }
                    }

                    return tile1.shape < tile2.shape;
                };

                std::set<TileInfo, decltype(isNCETileUnique)> uniqueShapeTiles(origTiles.begin(), origTiles.end(),
                                                                               isNCETileUnique);
                return OutputTiling(uniqueShapeTiles.begin(), uniqueShapeTiles.end());
            })
            .Case<VPU::NCEEltwiseOp>([&](VPU::NCEEltwiseOp) {
                std::set<TileInfo, TileShapeCompare> uniqueShapeTiles(origTiles.begin(), origTiles.end());
                OutputTiling outputTiles(uniqueShapeTiles.begin(), uniqueShapeTiles.end());
                return outputTiles;
            })
            .Default([&](mlir::Operation*) -> OutputTiling {
                return origTiles;
            });
}
}  // namespace VPU
}  // namespace vpux
