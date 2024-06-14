//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/common_rewriters/wrap_vpu_ops_in_ncecluster_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/numeric.hpp"

using namespace vpux;
using namespace VPU;

//
// NCEConvolutionRewriter
//

mlir::LogicalResult VPU::NCEConvolutionRewriter::matchAndRewrite(VPU::NCEConvolutionOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->template getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto* ctx = origOp->getContext();
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto strategy = origOp.getMultiClusterStrategy().value();

    auto outputTensorType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTensorType(
            clusteredOp, numClusters, strategy, outputTensorType, _enableExplicitDistributedTensorAttr);

    auto filterType = origOp.getFilter().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));
    const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTensorNumTiles =
            getIntArrayAttr(ctx, getWeightsTensorNumTiles(clusteredOp, filterType, numClusters.getInt(), strategy));
    const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTableTensorNumTiles = getIntArrayAttr(
            ctx, getWeightsTableTensorNumTiles(clusteredOp, outputType, numClusters.getInt(), strategy));

    const auto arch = VPU::getArch(origOp.getOperation());
    const auto canUseCMajor = VPU::NCEInvariant::isChannelMajorCompatible(
            arch, origOp.getInput().getType().cast<vpux::NDTypeInterface>());

    if (!canUseCMajor) {
        const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);

        if (activationAlignment.has_value()) {
            activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.value());
        }
    }

    const auto weightAlignment = getWeightsTensorAlignment(strategy);

    if (weightAlignment.has_value()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.value());
    }

    const auto distributedActivationCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getInput(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightsCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getFilter(), weightsTensorDistributionMode, weightsTensorNumTiles, weightAlignmentAttr,
            strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightTableCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getWeightsTable(), weightsTableTensorDistributionMode, weightsTableTensorNumTiles,
            weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::IRMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        if (newOp->hasAttr("multiClusterStrategy")) {
            newOp->removeAttr("multiClusterStrategy");
        }
        auto newOutput = newOp->getResult(0);
        const auto newOutType = newOutput.getType().cast<vpux::NDTypeInterface>();
        const auto cmxMemSpace = newOutType.changeMemSpace(MemoryKind::CMX_NN);
        newOutput.setType(cmxMemSpace);
        builder.create<YieldOp>(loc, newOp->getResults());
    };
    SmallVector<mlir::Value> distributedCopyOps{distributedActivationCopyOp->getResult(0),
                                                distributedWeightsCopyOp->getResult(0),
                                                distributedWeightTableCopyOp->getResult(0)};
    if (origOp.getActivationWindow() != nullptr) {
        const auto activationWindowDistributionMode = getActivationWindowTensorDistributionMode(strategy);
        const auto activationWindowNumTiles = getIntArrayAttr(ctx, getActivationWindowTensorNumTiles(strategy));
        auto distributedActivationWindowCopyOp = createDistributedCopyIn(
                clusteredOp, origOp.getActivationWindow(), activationWindowDistributionMode, activationWindowNumTiles,
                nullptr, strategy, _enableExplicitDistributedTensorAttr);

        distributedCopyOps.push_back(distributedActivationWindowCopyOp.getResult(0));
    }

    if (origOp.getInstructionListTable() != nullptr) {
        auto instructionListTableDistributionMode = getInstructionListTableTensorDistributionMode(strategy);
        auto instructionListTableNumTiles =
                getIntArrayAttr(origOp.getContext(), getInstructionListTableTensorNumTiles(strategy));
        auto distributedInstructionListTableCopyOp = createDistributedCopyIn(
                origOp, origOp.getInstructionListTable(), instructionListTableDistributionMode,
                instructionListTableNumTiles, nullptr, strategy, _enableExplicitDistributedTensorAttr);
        distributedCopyOps.push_back(distributedInstructionListTableCopyOp.getResult(0));
    }

    _log.trace("Wrap {0} into NCEClusterTilingOp", origOp->getName());

    const auto clusterTilingOp = rewriter.create<NCEClusterTilingOp>(origOp->getLoc(), distributedOutputTensorType,
                                                                     mlir::ValueRange{distributedCopyOps}, bodyBuilder);

    const auto outputCopyOp = createDistributedCopyOut(origOp, clusterTilingOp);
    const auto origOutput = origOp->getResult(0);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// NCEDepthConvolutionRewriter
//

mlir::LogicalResult VPU::NCEDepthConvolutionRewriter::matchAndRewrite(VPU::NCEDepthConvolutionOp origOp,
                                                                      mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto* ctx = origOp->getContext();
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    auto filterType = origOp.getFilter().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto strategy = origOp.getMultiClusterStrategy().value();

    auto outputTensorType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTensorType(
            clusteredOp, numClusters, strategy, outputTensorType, _enableExplicitDistributedTensorAttr);

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));
    const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTensorNumTiles =
            getIntArrayAttr(ctx, getWeightsTensorNumTiles(clusteredOp, filterType, numClusters.getInt(), strategy));
    const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTableTensorNumTiles = getIntArrayAttr(
            ctx, getWeightsTableTensorNumTiles(clusteredOp, outputType, numClusters.getInt(), strategy));

    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
    if (activationAlignment.has_value()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.value());
    }

    const auto weightAlignment = getWeightsTensorAlignment(strategy);

    if (weightAlignment.has_value()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.value());
    }

    const auto distributedActivationCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getInput(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightsCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getFilter(), weightsTensorDistributionMode, weightsTensorNumTiles, weightAlignmentAttr,
            strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightTableCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getWeightsTable(), weightsTableTensorDistributionMode, weightsTableTensorNumTiles,
            weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto origOutput = origOp->getResult(0);

    SmallVector<mlir::Value> distributedCopyOps{distributedActivationCopyOp->getResult(0),
                                                distributedWeightsCopyOp->getResult(0),
                                                distributedWeightTableCopyOp->getResult(0)};

    if (origOp.getActivationWindow() != nullptr) {
        const auto activationWindowDistributionMode = getActivationWindowTensorDistributionMode(strategy);
        const auto activationWindowNumTiles = getIntArrayAttr(ctx, getActivationWindowTensorNumTiles(strategy));
        const auto distributedActivationWindowCopyOp = createDistributedCopyIn(
                clusteredOp, origOp.getActivationWindow(), activationWindowDistributionMode, activationWindowNumTiles,
                nullptr, strategy, _enableExplicitDistributedTensorAttr);
        distributedCopyOps.push_back(distributedActivationWindowCopyOp->getResult(0));
    }

    if (origOp.getInstructionListTable() != nullptr) {
        auto instructionListTableDistributionMode = getInstructionListTableTensorDistributionMode(strategy);
        auto instructionListTableNumTiles =
                getIntArrayAttr(origOp.getContext(), getInstructionListTableTensorNumTiles(strategy));
        auto distributedInstructionListTableCopyOp = createDistributedCopyIn(
                origOp, origOp.getInstructionListTable(), instructionListTableDistributionMode,
                instructionListTableNumTiles, nullptr, strategy, _enableExplicitDistributedTensorAttr);
        distributedCopyOps.push_back(distributedInstructionListTableCopyOp.getResult(0));
    }

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::IRMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        if (newOp->hasAttr("multiClusterStrategy")) {
            newOp->removeAttr("multiClusterStrategy");
        }
        auto newOutput = newOp->getResult(0);
        const auto newOutType = newOutput.getType().cast<vpux::NDTypeInterface>();
        const auto cmxMemSpace = newOutType.changeMemSpace(MemoryKind::CMX_NN);
        newOutput.setType(cmxMemSpace);
        builder.create<YieldOp>(loc, newOp->getResults());
    };

    _log.trace("Wrap {0} into NCEClusterTilingOp", origOp->getName());
    const auto clusterTilingOp = rewriter.create<NCEClusterTilingOp>(origOp->getLoc(), distributedOutputTensorType,
                                                                     mlir::ValueRange{distributedCopyOps}, bodyBuilder);

    const auto outputCopyOp = createDistributedCopyOut(clusteredOp, clusterTilingOp);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// NCEMaxPoolRewriter
//

mlir::LogicalResult VPU::NCEMaxPoolRewriter::matchAndRewrite(VPU::NCEMaxPoolOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto* ctx = origOp->getContext();
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto strategy = origOp.getMultiClusterStrategy().value();

    auto outputTensorType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTensorType(
            clusteredOp, numClusters, strategy, outputTensorType, _enableExplicitDistributedTensorAttr);

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));

    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
    if (activationAlignment.has_value()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.value());
    }

    const auto weightAlignment = getWeightsTensorAlignment(strategy);
    if (weightAlignment.has_value()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.value());
    }

    const auto distributedActivationCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getInput(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    SmallVector<mlir::Value> distributedCopyOps{distributedActivationCopyOp->getResult(0)};

    if (origOp.getWeightsTable() != nullptr) {
        const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
        const auto weightsTableTensorNumTiles = getIntArrayAttr(
                ctx, getWeightsTableTensorNumTiles(clusteredOp, outputType, numClusters.getInt(), strategy));

        const auto distributedWeightTableCopyOp = createDistributedCopyIn(
                clusteredOp, origOp.getWeightsTable(), weightsTableTensorDistributionMode, weightsTableTensorNumTiles,
                weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);
        distributedCopyOps.push_back(distributedWeightTableCopyOp->getResult(0));
    }

    if (origOp.getActivationWindow() != nullptr) {
        const auto activationWindowDistributionMode = getActivationWindowTensorDistributionMode(strategy);
        const auto activationWindowNumTiles = getIntArrayAttr(ctx, getActivationWindowTensorNumTiles(strategy));

        const auto distributedActivationWindowCopyOp = createDistributedCopyIn(
                clusteredOp, origOp.getActivationWindow(), activationWindowDistributionMode, activationWindowNumTiles,
                nullptr, strategy, _enableExplicitDistributedTensorAttr);
        distributedCopyOps.push_back(distributedActivationWindowCopyOp->getResult(0));
    }
    const auto origOutput = origOp->getResult(0);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::IRMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        if (newOp->hasAttr("multiClusterStrategy")) {
            newOp->removeAttr("multiClusterStrategy");
        }
        auto newOutput = newOp->getResult(0);
        const auto newOutType = newOutput.getType().cast<vpux::NDTypeInterface>();
        const auto cmxMemSpace = newOutType.changeMemSpace(MemoryKind::CMX_NN);
        newOutput.setType(cmxMemSpace);
        builder.create<YieldOp>(loc, newOp->getResults());
    };

    const auto clusterTilingOp = rewriter.create<NCEClusterTilingOp>(origOp->getLoc(), distributedOutputTensorType,
                                                                     mlir::ValueRange{distributedCopyOps}, bodyBuilder);

    const auto outputCopyOp = createDistributedCopyOut(clusteredOp, clusterTilingOp);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// NCEAveragePoolRewriter
//

mlir::LogicalResult VPU::NCEAveragePoolRewriter::matchAndRewrite(VPU::NCEAveragePoolOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto* ctx = origOp->getContext();
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    const auto strategy = origOp.getMultiClusterStrategy().value();

    auto outputTensorType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTensorType(
            clusteredOp, numClusters, strategy, outputTensorType, _enableExplicitDistributedTensorAttr);

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));

    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
    if (activationAlignment.has_value()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.value());
    }

    const auto distributedActivationCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getInput(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto origOutput = origOp->getResult(0);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::IRMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        if (newOp->hasAttr("multiClusterStrategy")) {
            newOp->removeAttr("multiClusterStrategy");
        }
        auto newOutput = newOp->getResult(0);
        const auto newOutType = newOutput.getType().cast<vpux::NDTypeInterface>();
        const auto cmxMemSpace = newOutType.changeMemSpace(MemoryKind::CMX_NN);
        newOutput.setType(cmxMemSpace);
        builder.create<YieldOp>(loc, newOp->getResults());
    };

    const auto clusterTilingOp = rewriter.create<NCEClusterTilingOp>(
            origOp->getLoc(), distributedOutputTensorType, mlir::ValueRange{distributedActivationCopyOp->getResult(0)},
            bodyBuilder);

    const auto outputCopyOp = createDistributedCopyOut(clusteredOp, clusterTilingOp);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// NCEEltwiseRewriter
//

mlir::LogicalResult VPU::NCEEltwiseRewriter::matchAndRewrite(VPU::NCEEltwiseOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto* ctx = origOp->getContext();
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    const auto strategy = origOp.getMultiClusterStrategy().value();
    auto outputTensorType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTensorType(
            clusteredOp, numClusters, strategy, outputTensorType, _enableExplicitDistributedTensorAttr);

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));

    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
    if (activationAlignment.has_value()) {
        activationAlignmentAttr = getIntArrayAttr(origOp.getContext(), activationAlignment.value());
    }

    SmallVector<mlir::Value> newEltwiseInputs;
    if (origOp.getInput1() == origOp.getInput2()) {
        const auto distributedActivationCopyOp = createDistributedCopyIn(
                clusteredOp, origOp.getInput1(), activationTensorDistributionMode, activationTensorNumTiles,
                activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);
        newEltwiseInputs.push_back(distributedActivationCopyOp->getResult(0));
    } else {
        const auto distributedType1 = getDistributedTypeFromInput(
                clusteredOp, origOp.getInput1(), activationTensorDistributionMode, activationTensorNumTiles,
                activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

        const auto distributedType2 = getDistributedTypeFromInput(
                clusteredOp, origOp.getInput2(), activationTensorDistributionMode, activationTensorNumTiles,
                activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

        mlir::OpBuilder builder(clusteredOp);
        builder.setInsertionPoint(clusteredOp);
        const auto inputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                                mlir::ValueRange newOperands) {
            const auto memSpace = IndexedSymbolAttr::get(builder.getContext(), stringifyEnum(MemoryKind::CMX_NN));
            auto inputTensorDistributedCopyOp =
                    builder.create<VPU::CopyOp>(clusteredOp->getLoc(), newOperands[0], memSpace);
            builder.create<YieldOp>(loc, inputTensorDistributedCopyOp->getResults());
        };

        const auto distributedActivationCopyOp1 = builder.create<NCEClusterTilingOp>(
                clusteredOp->getLoc(), distributedType1, origOp.getInput1(), inputTensorBodyBuilder);
        const auto distributedActivationCopyOp2 = builder.create<NCEClusterTilingOp>(
                clusteredOp->getLoc(), distributedType2, origOp.getInput2(), inputTensorBodyBuilder);
        newEltwiseInputs.push_back(distributedActivationCopyOp1->getResult(0));
        newEltwiseInputs.push_back(distributedActivationCopyOp2->getResult(0));
    }

    const auto origOutput = origOp->getResult(0);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::IRMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        if (newOp->hasAttr("multiClusterStrategy")) {
            newOp->removeAttr("multiClusterStrategy");
        }
        auto newOutput = newOp->getResult(0);
        const auto newOutType = newOutput.getType().cast<vpux::NDTypeInterface>();
        const auto cmxMemSpace = newOutType.changeMemSpace(MemoryKind::CMX_NN);
        newOutput.setType(cmxMemSpace);
        builder.create<YieldOp>(loc, newOp->getResults());
    };

    const auto clusterTilingOp = rewriter.create<NCEClusterTilingOp>(origOp->getLoc(), distributedOutputTensorType,
                                                                     newEltwiseInputs, bodyBuilder);

    const auto outputCopyOp = createDistributedCopyOut(clusteredOp, clusterTilingOp);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// NCESWRewriter
//

mlir::LogicalResult VPU::NCESWRewriter::matchAndRewrite(VPU::SWOpInterface swOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), swOp->getName(), swOp->getLoc());

    if (swOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, swOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(swOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface", swOp);

    auto* ctx = swOp->getContext();

    const auto strategy = clusteredOp.getMultiClusterStrategy().value();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, getShape(swOp->getResult(0))[Dims4D::Act::C], strategy);

    SmallVector<mlir::Value> distributedCopyOps;
    for (auto operand : swOp->getOperands()) {
        const auto operandType = operand.getType().cast<vpux::NDTypeInterface>();
        const auto activationTensorDistributionMode =
                getSWInputTensorDistributionMode(clusteredOp, strategy, operandType);
        const auto activationTensorNumTiles = getIntArrayAttr(
                ctx, getSWInputTensorNumTiles(clusteredOp, numClusters.getInt(), strategy, operandType));

        // Input alignment is possibly needed to keep compatibility and avoid spilling
        // Only support:
        //       NCE_DPU (non SOH/SOHOverlapped)
        //          |
        //       NCE_SW  (Clustering/SOK)
        const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy, operandType);
        const auto activationAlignmentAttr = activationAlignment.has_value()
                                                     ? getIntArrayAttr(swOp.getContext(), activationAlignment.value())
                                                     : nullptr;

        const auto distributedCopyOp = createDistributedCopyIn(clusteredOp, operand, activationTensorDistributionMode,
                                                               activationTensorNumTiles, activationAlignmentAttr,
                                                               strategy, _enableExplicitDistributedTensorAttr);
        distributedCopyOps.push_back(distributedCopyOp->getResult(0));
    }

    SmallVector<mlir::Type> distributedOutputTypes;
    for (const auto& origOutput : swOp->getResults()) {
        _log.trace("[{0}] Got tag: {1}\n", getDebugName(), origOutput);
        auto outputTensorType = origOutput.getType().cast<vpux::NDTypeInterface>();
        // Output alignment is possibly needed to keep compatibility and avoid spilling
        // Only support:
        //       NCE_SW  (Clustering/SOK)
        //          |
        //       NCE_DPU (non SOH/SOHOverlapped)
        auto distributedOutputTensorType = getDistributedOutputTensorType(
                clusteredOp, numClusters, strategy, outputTensorType, _enableExplicitDistributedTensorAttr,
                /*alignForSOH=*/false);
        distributedOutputTypes.push_back(distributedOutputTensorType);
    }

    const auto bodyBuilder = [swOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        auto* newOp = builder.clone(*swOp);
        if (newOp->hasAttr("multiClusterStrategy")) {
            newOp->removeAttr("multiClusterStrategy");
        }
        for (auto operand : swOp->getOperands() | indexed) {
            newOp->setOperand(operand.index(), newOperands[operand.index()]);
        }

        for (const auto& result : swOp->getResults() | indexed) {
            auto newOutput = newOp->getResult(result.index());
            const auto newOutType = newOutput.getType().cast<vpux::NDTypeInterface>();
            const auto cmxMemSpace = newOutType.changeMemSpace(MemoryKind::CMX_NN);
            newOutput.setType(cmxMemSpace);
        }

        builder.create<YieldOp>(loc, newOp->getResults());
    };

    const auto clusterTilingOp = rewriter.create<NCEClusterTilingOp>(
            swOp->getLoc(), mlir::TypeRange{distributedOutputTypes}, mlir::ValueRange{distributedCopyOps}, bodyBuilder);

    SmallVector<mlir::Value> newOutputs;
    for (const auto& result : swOp->getResults() | indexed) {
        const auto index = result.index();
        const auto origOutput = result.value();
        const auto origOutType = origOutput.getType().cast<NDTypeInterface>();
        const auto origOutMemSpace = origOutType.getMemSpace();

        const auto outputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                                 mlir::ValueRange newOperands) {
            auto outputTensorDistributedCopyOp = builder.create<VPU::CopyOp>(loc, newOperands[0], origOutMemSpace);
            builder.create<YieldOp>(loc, outputTensorDistributedCopyOp->getResults());
        };

        auto outputCopyOp = rewriter.create<NCEClusterTilingOp>(
                clusterTilingOp->getLoc(), origOutType, clusterTilingOp->getResult(index), outputTensorBodyBuilder);

        origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
        newOutputs.push_back(outputCopyOp->getResult(0));
    }

    rewriter.replaceOp(swOp, newOutputs);
    return mlir::success();
}

//
// NCECompressConvolutionRewriter
//

mlir::LogicalResult VPU::NCECompressConvolutionRewriter::matchAndRewrite(VPU::NCECompressConvolutionOp origOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->template getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto* ctx = origOp->getContext();
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto strategy = origOp.getMultiClusterStrategy().value();

    auto outputTensorType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTensorType(
            clusteredOp, numClusters, strategy, outputTensorType, _enableExplicitDistributedTensorAttr);

    auto filterType = origOp.getFilter().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));
    const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTensorNumTiles =
            getIntArrayAttr(ctx, getWeightsTensorNumTiles(clusteredOp, filterType, numClusters.getInt(), strategy));
    const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTableTensorNumTiles = getIntArrayAttr(
            ctx, getWeightsTableTensorNumTiles(clusteredOp, outputType, numClusters.getInt(), strategy));

    const auto weightAlignment = getWeightsTensorAlignment(strategy);

    if (weightAlignment.has_value()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.value());
    }

    const auto distributedActivationCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getInput(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightsCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getFilter(), weightsTensorDistributionMode, weightsTensorNumTiles, weightAlignmentAttr,
            strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightTableCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getWeightsTable(), weightsTableTensorDistributionMode, weightsTableTensorNumTiles,
            weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::IRMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        if (newOp->hasAttr("multiClusterStrategy")) {
            newOp->removeAttr("multiClusterStrategy");
        }
        auto newOutput = newOp->getResult(0);
        const auto newOutType = newOutput.getType().cast<vpux::NDTypeInterface>();
        const auto cmxMemSpace = newOutType.changeMemSpace(MemoryKind::CMX_NN);
        newOutput.setType(cmxMemSpace);
        builder.create<YieldOp>(loc, newOp->getResults());
    };
    SmallVector<mlir::Value> distributedCopyOps{distributedActivationCopyOp->getResult(0),
                                                distributedWeightsCopyOp->getResult(0),
                                                distributedWeightTableCopyOp->getResult(0)};

    _log.trace("Wrap {0} into NCEClusterTilingOp", origOp->getName());

    const auto clusterTilingOp = rewriter.create<NCEClusterTilingOp>(origOp->getLoc(), distributedOutputTensorType,
                                                                     mlir::ValueRange{distributedCopyOps}, bodyBuilder);

    const auto outputCopyOp = createDistributedCopyOut(origOp, clusterTilingOp);
    const auto origOutput = origOp->getResult(0);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

//
// NCEInterpolateRewriter
//

mlir::LogicalResult VPU::NCEInterpolateRewriter::matchAndRewrite(VPU::NCEInterpolateOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->template getParentOfType<NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    auto* ctx = origOp->getContext();
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    const auto strategy = origOp.getMultiClusterStrategy().value();

    auto outputTensorType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTensorType(
            clusteredOp, numClusters, strategy, outputTensorType, _enableExplicitDistributedTensorAttr);

    auto weightsType = origOp.getWeights().getType().cast<vpux::NDTypeInterface>();
    auto outputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));
    const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTensorNumTiles =
            getIntArrayAttr(ctx, getWeightsTensorNumTiles(clusteredOp, weightsType, numClusters.getInt(), strategy));
    const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTableTensorNumTiles = getIntArrayAttr(
            ctx, getWeightsTableTensorNumTiles(clusteredOp, outputType, numClusters.getInt(), strategy));

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
    if (activationAlignment.has_value()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.value());
    }

    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto weightAlignment = getWeightsTensorAlignment(strategy);
    if (weightAlignment.has_value()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.value());
    }

    const auto distributedActivationCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getInput(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightsCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getWeights(), weightsTensorDistributionMode, weightsTensorNumTiles, weightAlignmentAttr,
            strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightTableCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getWeightsTable(), weightsTableTensorDistributionMode, weightsTableTensorNumTiles,
            weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::IRMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        if (newOp->hasAttr("multiClusterStrategy")) {
            newOp->removeAttr("multiClusterStrategy");
        }
        auto newOutput = newOp->getResult(0);
        const auto newOutType = newOutput.getType().cast<vpux::NDTypeInterface>();
        const auto cmxMemSpace = newOutType.changeMemSpace(MemoryKind::CMX_NN);
        newOutput.setType(cmxMemSpace);
        builder.create<YieldOp>(loc, newOp->getResults());
    };
    SmallVector<mlir::Value> distributedCopyOps{distributedActivationCopyOp->getResult(0),
                                                distributedWeightsCopyOp->getResult(0),
                                                distributedWeightTableCopyOp->getResult(0)};

    _log.trace("Wrap {0} into NCEClusterTilingOp", origOp->getName());

    const auto clusterTilingOp = rewriter.create<NCEClusterTilingOp>(origOp->getLoc(), distributedOutputTensorType,
                                                                     mlir::ValueRange{distributedCopyOps}, bodyBuilder);

    const auto outputCopyOp = createDistributedCopyOut(origOp, clusterTilingOp);
    const auto origOutput = origOp->getResult(0);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}
