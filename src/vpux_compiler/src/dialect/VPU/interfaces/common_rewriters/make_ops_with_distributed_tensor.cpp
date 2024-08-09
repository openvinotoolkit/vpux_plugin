//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/common_rewriters/make_ops_with_distributed_tensor.hpp"
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
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape(), strategy);
    auto overlappedParams = _overlapParamsLookup.at(origOp->getResult(0));
    vpux::NDTypeInterface distributedOutputTensorType =
            getDistributedOutputTensorType(clusteredOp, numClusters, strategy, outputTensorType,
                                           _enableExplicitDistributedTensorAttr, true, overlappedParams);

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

    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters.getInt(), strategy);

    if (activationAlignment.has_value()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.value());
    }

    const auto weightAlignment = getWeightsTensorAlignment(strategy);

    if (weightAlignment.has_value()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.value());
    }

    const auto distributedActivationCopyOp = createDistributedCopyIn(
            rewriter, clusteredOp, origOp.getInput(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightsCopyOp = createDistributedCopyIn(
            rewriter, clusteredOp, origOp.getFilter(), weightsTensorDistributionMode, weightsTensorNumTiles,
            weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightTableCopyOp = createDistributedCopyIn(
            rewriter, clusteredOp, origOp.getWeightsTable(), weightsTableTensorDistributionMode,
            weightsTableTensorNumTiles, weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    SmallVector<mlir::Value> distributedCopyOps{distributedActivationCopyOp->getResult(0),
                                                distributedWeightsCopyOp->getResult(0),
                                                distributedWeightTableCopyOp->getResult(0)};
    if (origOp.getActivationWindow() != nullptr) {
        const auto activationWindowDistributionMode = getActivationWindowTensorDistributionMode(strategy);
        const auto activationWindowNumTiles = getIntArrayAttr(ctx, getActivationWindowTensorNumTiles(strategy));
        auto distributedActivationWindowCopyOp = createDistributedCopyIn(
                rewriter, clusteredOp, origOp.getActivationWindow(), activationWindowDistributionMode,
                activationWindowNumTiles, nullptr, strategy, _enableExplicitDistributedTensorAttr);

        distributedCopyOps.push_back(distributedActivationWindowCopyOp.getResult());
    }

    if (origOp.getInstructionListTable() != nullptr) {
        auto instructionListTableDistributionMode = getInstructionListTableTensorDistributionMode(strategy);
        auto instructionListTableNumTiles =
                getIntArrayAttr(origOp.getContext(), getInstructionListTableTensorNumTiles(strategy));
        auto distributedInstructionListTableCopyOp = createDistributedCopyIn(
                rewriter, origOp, origOp.getInstructionListTable(), instructionListTableDistributionMode,
                instructionListTableNumTiles, nullptr, strategy, _enableExplicitDistributedTensorAttr);
        distributedCopyOps.push_back(distributedInstructionListTableCopyOp.getResult());
    }

    mlir::IRMapping mapper;
    mapper.map(origOp->getOperands(), mlir::ValueRange{distributedCopyOps});
    auto* newOp = rewriter.clone(*origOp, mapper);
    newOp->getResult(0).setType(distributedOutputTensorType);

    if (newOp->hasAttr(multiClusterStrategy)) {
        newOp->removeAttr(multiClusterStrategy);
    }
    auto outputCopyOp =
            rewriter.create<VPU::CopyOp>(newOp->getLoc(), origOp->getResult(0).getType(), newOp->getResult(0), nullptr);

    rewriter.replaceOp(origOp, outputCopyOp);

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
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape(), strategy);
    auto overlappedParams = _overlapParamsLookup.at(origOp->getResult(0));
    vpux::NDTypeInterface distributedOutputTensorType =
            getDistributedOutputTensorType(clusteredOp, numClusters, strategy, outputTensorType,
                                           _enableExplicitDistributedTensorAttr, true, overlappedParams);

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));
    const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTensorNumTiles =
            getIntArrayAttr(ctx, getWeightsTensorNumTiles(clusteredOp, filterType, numClusters.getInt(), strategy));
    const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
    const auto weightsTableTensorNumTiles = getIntArrayAttr(
            ctx, getWeightsTableTensorNumTiles(clusteredOp, outputType, numClusters.getInt(), strategy));

    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters.getInt(), strategy);
    if (activationAlignment.has_value()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.value());
    }

    const auto weightAlignment = getWeightsTensorAlignment(strategy);

    if (weightAlignment.has_value()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.value());
    }

    const auto distributedActivationCopyOp = createDistributedCopyIn(
            rewriter, clusteredOp, origOp.getInput(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightsCopyOp = createDistributedCopyIn(
            rewriter, clusteredOp, origOp.getFilter(), weightsTensorDistributionMode, weightsTensorNumTiles,
            weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightTableCopyOp = createDistributedCopyIn(
            rewriter, clusteredOp, origOp.getWeightsTable(), weightsTableTensorDistributionMode,
            weightsTableTensorNumTiles, weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    SmallVector<mlir::Value> distributedCopyOps{distributedActivationCopyOp->getResult(0),
                                                distributedWeightsCopyOp->getResult(0),
                                                distributedWeightTableCopyOp->getResult(0)};

    if (origOp.getActivationWindow() != nullptr) {
        const auto activationWindowDistributionMode = getActivationWindowTensorDistributionMode(strategy);
        const auto activationWindowNumTiles = getIntArrayAttr(ctx, getActivationWindowTensorNumTiles(strategy));
        const auto distributedActivationWindowCopyOp = createDistributedCopyIn(
                rewriter, clusteredOp, origOp.getActivationWindow(), activationWindowDistributionMode,
                activationWindowNumTiles, nullptr, strategy, _enableExplicitDistributedTensorAttr);
        distributedCopyOps.push_back(distributedActivationWindowCopyOp->getResult(0));
    }

    if (origOp.getInstructionListTable() != nullptr) {
        auto instructionListTableDistributionMode = getInstructionListTableTensorDistributionMode(strategy);
        auto instructionListTableNumTiles =
                getIntArrayAttr(origOp.getContext(), getInstructionListTableTensorNumTiles(strategy));
        auto distributedInstructionListTableCopyOp = createDistributedCopyIn(
                rewriter, origOp, origOp.getInstructionListTable(), instructionListTableDistributionMode,
                instructionListTableNumTiles, nullptr, strategy, _enableExplicitDistributedTensorAttr);
        distributedCopyOps.push_back(distributedInstructionListTableCopyOp.getResult());
    }

    mlir::IRMapping mapper;
    mapper.map(origOp->getOperands(), mlir::ValueRange{distributedCopyOps});
    auto* newOp = rewriter.clone(*origOp, mapper);
    newOp->getResult(0).setType(distributedOutputTensorType);
    if (newOp->hasAttr(multiClusterStrategy)) {
        newOp->removeAttr(multiClusterStrategy);
    }
    auto outputCopyOp =
            rewriter.create<VPU::CopyOp>(newOp->getLoc(), origOp->getResult(0).getType(), newOp->getResult(0), nullptr);

    rewriter.replaceOp(origOp, outputCopyOp);

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
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape(), strategy);
    auto overlappedParams = _overlapParamsLookup.at(origOp->getResult(0));
    vpux::NDTypeInterface distributedOutputTensorType =
            getDistributedOutputTensorType(clusteredOp, numClusters, strategy, outputTensorType,
                                           _enableExplicitDistributedTensorAttr, true, overlappedParams);

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));

    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters.getInt(), strategy);
    if (activationAlignment.has_value()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.value());
    }

    const auto weightAlignment = getWeightsTensorAlignment(strategy);
    if (weightAlignment.has_value()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.value());
    }

    const auto distributedActivationCopyOp = createDistributedCopyIn(
            rewriter, clusteredOp, origOp.getInput(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    SmallVector<mlir::Value> distributedCopyOps{distributedActivationCopyOp->getResult(0)};

    if (origOp.getWeightsTable() != nullptr) {
        const auto weightsTableTensorDistributionMode = getWeightsTensorDistributionMode(strategy);
        const auto weightsTableTensorNumTiles = getIntArrayAttr(
                ctx, getWeightsTableTensorNumTiles(clusteredOp, outputType, numClusters.getInt(), strategy));

        const auto distributedWeightTableCopyOp = createDistributedCopyIn(
                rewriter, clusteredOp, origOp.getWeightsTable(), weightsTableTensorDistributionMode,
                weightsTableTensorNumTiles, weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);
        distributedCopyOps.push_back(distributedWeightTableCopyOp->getResult(0));
    }

    if (origOp.getActivationWindow() != nullptr) {
        const auto activationWindowDistributionMode = getActivationWindowTensorDistributionMode(strategy);
        const auto activationWindowNumTiles = getIntArrayAttr(ctx, getActivationWindowTensorNumTiles(strategy));

        const auto distributedActivationWindowCopyOp = createDistributedCopyIn(
                rewriter, clusteredOp, origOp.getActivationWindow(), activationWindowDistributionMode,
                activationWindowNumTiles, nullptr, strategy, _enableExplicitDistributedTensorAttr);
        distributedCopyOps.push_back(distributedActivationWindowCopyOp->getResult(0));
    }

    mlir::IRMapping mapper;
    mapper.map(origOp->getOperands(), mlir::ValueRange{distributedCopyOps});
    auto* newOp = rewriter.clone(*origOp, mapper);
    newOp->getResult(0).setType(distributedOutputTensorType);
    if (newOp->hasAttr(multiClusterStrategy)) {
        newOp->removeAttr(multiClusterStrategy);
    }
    auto outputCopyOp =
            rewriter.create<VPU::CopyOp>(newOp->getLoc(), origOp->getResult(0).getType(), newOp->getResult(0), nullptr);

    rewriter.replaceOp(origOp, outputCopyOp);

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
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape(), strategy);
    auto overlappedParams = _overlapParamsLookup.at(origOp->getResult(0));
    vpux::NDTypeInterface distributedOutputTensorType =
            getDistributedOutputTensorType(clusteredOp, numClusters, strategy, outputTensorType,
                                           _enableExplicitDistributedTensorAttr, true, overlappedParams);

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));

    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters.getInt(), strategy);
    if (activationAlignment.has_value()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.value());
    }

    const auto distributedActivationCopyOp = createDistributedCopyIn(
            rewriter, clusteredOp, origOp.getInput(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    SmallVector<mlir::Value> distributedCopyOps{distributedActivationCopyOp->getResult(0)};

    mlir::IRMapping mapper;
    mapper.map(origOp->getOperands(), mlir::ValueRange{distributedCopyOps});
    auto* newOp = rewriter.clone(*origOp, mapper);
    newOp->getResult(0).setType(distributedOutputTensorType);
    if (newOp->hasAttr(multiClusterStrategy)) {
        newOp->removeAttr(multiClusterStrategy);
    }
    auto outputCopyOp =
            rewriter.create<VPU::CopyOp>(newOp->getLoc(), origOp->getResult(0).getType(), newOp->getResult(0), nullptr);

    rewriter.replaceOp(origOp, outputCopyOp);

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
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape(), strategy);
    auto overlappedParams = _overlapParamsLookup.at(origOp->getResult(0));
    vpux::NDTypeInterface distributedOutputTensorType =
            getDistributedOutputTensorType(clusteredOp, numClusters, strategy, outputTensorType,
                                           _enableExplicitDistributedTensorAttr, true, overlappedParams);

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));

    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters.getInt(), strategy);
    if (activationAlignment.has_value()) {
        activationAlignmentAttr = getIntArrayAttr(origOp.getContext(), activationAlignment.value());
    }

    SmallVector<mlir::Value> distributedCopyOps;
    if (origOp.getInput1() == origOp.getInput2()) {
        const auto distributedActivationCopyOp = createDistributedCopyIn(
                rewriter, clusteredOp, origOp.getInput1(), activationTensorDistributionMode, activationTensorNumTiles,
                activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);
        distributedCopyOps.push_back(distributedActivationCopyOp->getResult(0));
    } else {
        const auto distributedType1 = getDistributedTypeFromInput(
                clusteredOp, origOp.getInput1(), activationTensorDistributionMode, activationTensorNumTiles,
                activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

        const auto distributedType2 = getDistributedTypeFromInput(
                clusteredOp, origOp.getInput2(), activationTensorDistributionMode, activationTensorNumTiles,
                activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

        rewriter.setInsertionPoint(clusteredOp);
        const auto memSpace = IndexedSymbolAttr::get(rewriter.getContext(), stringifyEnum(MemoryKind::CMX_NN));
        auto distributedActivationCopyOp1 =
                rewriter.create<VPU::CopyOp>(clusteredOp.getLoc(), distributedType1, origOp.getInput1(), memSpace);
        auto distributedActivationCopyOp2 =
                rewriter.create<VPU::CopyOp>(clusteredOp.getLoc(), distributedType2, origOp.getInput2(), memSpace);

        distributedCopyOps.push_back(distributedActivationCopyOp1->getResult(0));
        distributedCopyOps.push_back(distributedActivationCopyOp2->getResult(0));
    }

    mlir::IRMapping mapper;
    mapper.map(origOp->getOperands(), mlir::ValueRange{distributedCopyOps});
    auto* newOp = rewriter.clone(*origOp, mapper);
    newOp->getResult(0).setType(distributedOutputTensorType);
    if (newOp->hasAttr(multiClusterStrategy)) {
        newOp->removeAttr(multiClusterStrategy);
    }
    auto outputCopyOp =
            rewriter.create<VPU::CopyOp>(newOp->getLoc(), origOp->getResult(0).getType(), newOp->getResult(0), nullptr);

    rewriter.replaceOp(origOp, outputCopyOp);

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
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, getShape(swOp->getResult(0)), strategy);

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
        const auto activationAlignment =
                getActivationTensorAlignment(clusteredOp, numClusters.getInt(), strategy, operandType);
        const auto activationAlignmentAttr = activationAlignment.has_value()
                                                     ? getIntArrayAttr(swOp.getContext(), activationAlignment.value())
                                                     : nullptr;

        const auto distributedCopyOp = createDistributedCopyIn(
                rewriter, clusteredOp, operand, activationTensorDistributionMode, activationTensorNumTiles,
                activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);
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
        auto overlappedParams = _overlapParamsLookup.at(origOutput);
        vpux::NDTypeInterface distributedOutputTensorType =
                getDistributedOutputTensorType(clusteredOp, numClusters, strategy, outputTensorType,
                                               _enableExplicitDistributedTensorAttr, true, overlappedParams);
        distributedOutputTypes.push_back(distributedOutputTensorType);
    }
    // Replace the operands manually, as a mapper would have an issue with identical inputs
    auto* newOp = rewriter.clone(*swOp);
    for (auto operand : swOp->getOperands() | indexed) {
        newOp->setOperand(operand.index(), distributedCopyOps[operand.index()]);
    }

    SmallVector<mlir::Value> newCopyOutputs;
    if (newOp->hasAttr(multiClusterStrategy)) {
        newOp->removeAttr(multiClusterStrategy);
    }

    rewriter.setInsertionPointAfter(newOp);
    for (auto result : newOp->getResults() | indexed) {
        result.value().setType(distributedOutputTypes[result.index()]);
        auto outputCopyOp = rewriter.create<VPU::CopyOp>(newOp->getLoc(), swOp->getResult(result.index()).getType(),
                                                         result.value(), nullptr);
        newCopyOutputs.push_back(outputCopyOp->getResult(0));
    }

    rewriter.replaceOp(swOp, newCopyOutputs);

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
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape(), strategy);
    auto overlappedParams = _overlapParamsLookup.at(origOp->getResult(0));
    vpux::NDTypeInterface distributedOutputTensorType =
            getDistributedOutputTensorType(clusteredOp, numClusters, strategy, outputTensorType,
                                           _enableExplicitDistributedTensorAttr, true, overlappedParams);

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
            rewriter, clusteredOp, origOp.getInput(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightsCopyOp = createDistributedCopyIn(
            rewriter, clusteredOp, origOp.getFilter(), weightsTensorDistributionMode, weightsTensorNumTiles,
            weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightTableCopyOp = createDistributedCopyIn(
            rewriter, clusteredOp, origOp.getWeightsTable(), weightsTableTensorDistributionMode,
            weightsTableTensorNumTiles, weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    SmallVector<mlir::Value> distributedCopyOps{distributedActivationCopyOp->getResult(0),
                                                distributedWeightsCopyOp->getResult(0),
                                                distributedWeightTableCopyOp->getResult(0)};

    mlir::IRMapping mapper;
    mapper.map(origOp->getOperands(), mlir::ValueRange{distributedCopyOps});
    auto* newOp = rewriter.clone(*origOp, mapper);
    if (newOp->hasAttr(multiClusterStrategy)) {
        newOp->removeAttr(multiClusterStrategy);
    }
    newOp->getResult(0).setType(distributedOutputTensorType);
    auto outputCopyOp =
            rewriter.create<VPU::CopyOp>(newOp->getLoc(), origOp->getResult(0).getType(), newOp->getResult(0), nullptr);

    rewriter.replaceOp(origOp, outputCopyOp);

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
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape(), strategy);
    auto overlappedParams = _overlapParamsLookup.at(origOp->getResult(0));
    vpux::NDTypeInterface distributedOutputTensorType =
            getDistributedOutputTensorType(clusteredOp, numClusters, strategy, outputTensorType,
                                           _enableExplicitDistributedTensorAttr, true, overlappedParams);

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
    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters.getInt(), strategy);
    if (activationAlignment.has_value()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.value());
    }

    mlir::ArrayAttr weightAlignmentAttr = nullptr;
    const auto weightAlignment = getWeightsTensorAlignment(strategy);
    if (weightAlignment.has_value()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.value());
    }

    const auto distributedActivationCopyOp = createDistributedCopyIn(
            rewriter, clusteredOp, origOp.getInput(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightsCopyOp = createDistributedCopyIn(
            rewriter, clusteredOp, origOp.getWeights(), weightsTensorDistributionMode, weightsTensorNumTiles,
            weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightTableCopyOp = createDistributedCopyIn(
            rewriter, clusteredOp, origOp.getWeightsTable(), weightsTableTensorDistributionMode,
            weightsTableTensorNumTiles, weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    SmallVector<mlir::Value> distributedCopyOps{distributedActivationCopyOp->getResult(0),
                                                distributedWeightsCopyOp->getResult(0),
                                                distributedWeightTableCopyOp->getResult(0)};

    mlir::IRMapping mapper;
    mapper.map(origOp->getOperands(), mlir::ValueRange{distributedCopyOps});
    auto* newOp = rewriter.clone(*origOp, mapper);
    if (newOp->hasAttr(multiClusterStrategy)) {
        newOp->removeAttr(multiClusterStrategy);
    }
    newOp->getResult(0).setType(distributedOutputTensorType);
    auto outputCopyOp =
            rewriter.create<VPU::CopyOp>(newOp->getLoc(), origOp->getResult(0).getType(), newOp->getResult(0), nullptr);

    rewriter.replaceOp(origOp, outputCopyOp);

    return mlir::success();
}

//
// NCEMatMulRewriter
//

mlir::LogicalResult VPU::NCEMatMulRewriter::matchAndRewrite(VPU::NCEMatMulOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<VPU::NCEClusterTilingOp>() != nullptr) {
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
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape(), strategy);
    auto overlappedParams = _overlapParamsLookup.at(origOp->getResult(0));
    auto distributedOutputTensorType =
            getDistributedOutputTensorType(clusteredOp, numClusters, strategy, outputTensorType,
                                           _enableExplicitDistributedTensorAttr, true, overlappedParams);

    auto filterType = origOp.getWeights().getType().cast<vpux::NDTypeInterface>();
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

    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters.getInt(), strategy);

    if (activationAlignment.has_value()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.value());
    }

    const auto weightAlignment = getWeightsTensorAlignment(strategy);

    if (weightAlignment.has_value()) {
        weightAlignmentAttr = getIntArrayAttr(ctx, weightAlignment.value());
    }

    const auto distributedActivationCopyOp = createDistributedCopyIn(
            rewriter, clusteredOp, origOp.getInput(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightsCopyOp = createDistributedCopyIn(
            rewriter, clusteredOp, origOp.getWeights(), weightsTensorDistributionMode, weightsTensorNumTiles,
            weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    const auto distributedWeightTableCopyOp = createDistributedCopyIn(
            rewriter, clusteredOp, origOp.getWeightsTable(), weightsTableTensorDistributionMode,
            weightsTableTensorNumTiles, weightAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

    SmallVector<mlir::Value> distributedCopyOps{distributedActivationCopyOp->getResult(0),
                                                distributedWeightsCopyOp->getResult(0),
                                                distributedWeightTableCopyOp->getResult(0)};

    mlir::IRMapping mapper;
    mapper.map(origOp->getOperands(), mlir::ValueRange{distributedCopyOps});
    auto* newOp = rewriter.clone(*origOp, mapper);
    newOp->getResult(0).setType(distributedOutputTensorType);

    if (newOp->hasAttr(multiClusterStrategy)) {
        newOp->removeAttr(multiClusterStrategy);
    }
    auto outputCopyOp =
            rewriter.create<VPU::CopyOp>(newOp->getLoc(), origOp->getResult(0).getType(), newOp->getResult(0), nullptr);

    rewriter.replaceOp(origOp, outputCopyOp);

    return mlir::success();
}
