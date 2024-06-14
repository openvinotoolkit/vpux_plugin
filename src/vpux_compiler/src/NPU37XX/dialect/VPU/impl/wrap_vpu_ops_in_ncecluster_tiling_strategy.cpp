//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/wrap_vpu_ops_in_ncecluster_tiling_strategy.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/common_rewriters/wrap_vpu_ops_in_ncecluster_tiling.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"

using namespace vpux;

//
// NCEPermuteRewriter
//

namespace {

class NCEPermuteRewriter final : public mlir::OpRewritePattern<VPU::NCEPermuteOp> {
public:
    NCEPermuteRewriter(mlir::MLIRContext* ctx, bool enableExplicitDistributedTensorAttr, Logger log)
            : mlir::OpRewritePattern<VPU::NCEPermuteOp>(ctx),
              _enableExplicitDistributedTensorAttr(enableExplicitDistributedTensorAttr),
              _log(log) {
        setDebugName("NCEPermuteRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::NCEClusterTilingOp buildInputCopy(VPU::ClusteredOpInterface clusteredOp, mlir::Value input,
                                           mlir::Type distType) const;
    bool _enableExplicitDistributedTensorAttr = false;
    Logger _log;
};

mlir::LogicalResult NCEPermuteRewriter::matchAndRewrite(VPU::NCEPermuteOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (origOp->getParentOfType<VPU::NCEClusterTilingOp>() != nullptr) {
        return matchFailed(_log, rewriter, origOp, "The operation is already wrapped with NCEClusterTiling");
    }

    if (!origOp.getMultiClusterStrategy().has_value()) {
        return matchFailed(_log, rewriter, origOp, "The operation does not have multi-cluster strategy.");
    }

    auto* ctx = origOp->getContext();
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    const auto nextConv = getNextCompressConv(origOp);
    const auto strategy = nextConv == nullptr ? VPU::MultiClusterStrategy::SplitOverHeight
                                              : VPU::MultiClusterStrategy::SplitOverHeightOverlapped;

    auto outputTensorType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
    if (activationAlignment.has_value()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.value());
    }
    const auto neutralPads = VPU::getPaddingAttr(ctx, 0, 0, 0, 0);
    const auto overlapParams = VPU::OverlapDistributionParams(
            getIntArrayAttr(ctx, origOp.getKernelSizeVal()), neutralPads, getIntArrayAttr(ctx, origOp.getStridesVal()));

    auto inputTensorType = origOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto uniformDistributedSegments = VPU::getUniformDistributedSegments(
            clusteredOp, inputTensorType.getShape().raw(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr);
    const auto inputDistType = VPU::createDistributedTensorType(
            clusteredOp, inputTensorType, activationTensorDistributionMode, activationTensorNumTiles, numClusters,
            activationAlignmentAttr, uniformDistributedSegments, _enableExplicitDistributedTensorAttr, overlapParams);
    const auto fusedDistType =
            fuseOverlapParams(clusteredOp, inputDistType, nextConv, _enableExplicitDistributedTensorAttr);

    const auto inputCopyOp = buildInputCopy(clusteredOp, origOp.getInput(), fusedDistType);

    const auto distributedOutputTensorType =
            getDistributedOutputTypeFromOp(origOp, origOp->getResult(0).getType(), numClusters, strategy);

    const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::IRMapping mapper;
        mapper.map(origOp->getOperands(), newOperands);
        auto* newOp = builder.clone(*origOp, mapper);
        if (newOp->hasAttr("multiClusterStrategy")) {
            newOp->removeAttr("multiClusterStrategy");
        }
        auto newOutput = newOp->getResult(0);
        const auto newOutType = newOutput.getType().cast<vpux::NDTypeInterface>();
        const auto cmxMemSpace = newOutType.changeMemSpace(VPU::MemoryKind::CMX_NN);
        newOutput.setType(cmxMemSpace);
        builder.create<VPU::YieldOp>(loc, newOp->getResults());
    };

    _log.trace("Wrap {0} into NCEClusterTilingOp", origOp->getName());

    const auto clusterTilingOp = rewriter.create<VPU::NCEClusterTilingOp>(
            origOp->getLoc(), distributedOutputTensorType, mlir::ValueRange{inputCopyOp->getResult(0)}, bodyBuilder);

    const auto outputCopyOp = createDistributedCopyOut(origOp, clusterTilingOp);
    const auto origOutput = origOp->getResult(0);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

VPU::NCEClusterTilingOp NCEPermuteRewriter::buildInputCopy(VPU::ClusteredOpInterface clusteredOp, mlir::Value input,
                                                           mlir::Type distType) const {
    mlir::OpBuilder builder(clusteredOp);
    builder.setInsertionPoint(clusteredOp);
    const auto inputTensorBodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc,
                                            mlir::ValueRange newOperands) {
        const auto memSpace = IndexedSymbolAttr::get(builder.getContext(), stringifyEnum(VPU::MemoryKind::CMX_NN));
        auto inputTensorDistributedCopyOp =
                builder.create<VPU::CopyOp>(clusteredOp->getLoc(), newOperands[0], memSpace);
        builder.create<VPU::YieldOp>(loc, inputTensorDistributedCopyOp->getResults());
    };

    auto distributedInputCopyOp =
            builder.create<VPU::NCEClusterTilingOp>(clusteredOp->getLoc(), distType, input, inputTensorBodyBuilder);

    return distributedInputCopyOp;
}

}  // namespace

//
// WrapVPUOpsInNCEClusterTilingStrategy
//

void VPU::arch37xx::WrapVPUOpsInNCEClusterTilingStrategy::addPatterns(mlir::RewritePatternSet& patterns,
                                                                      Logger& log) const {
    auto ctx = patterns.getContext();
    patterns.add<VPU::NCEConvolutionRewriter>(ctx, _enableExplicitDistributedTensorAttr, log);
    patterns.add<VPU::NCEDepthConvolutionRewriter>(ctx, _enableExplicitDistributedTensorAttr, log);
    patterns.add<VPU::NCEMaxPoolRewriter>(ctx, _enableExplicitDistributedTensorAttr, log);
    patterns.add<VPU::NCEAveragePoolRewriter>(ctx, _enableExplicitDistributedTensorAttr, log);
    patterns.add<VPU::NCEEltwiseRewriter>(ctx, _enableExplicitDistributedTensorAttr, log);
    patterns.add<VPU::NCESWRewriter>(ctx, _enableExplicitDistributedTensorAttr, log);
    patterns.add<VPU::NCECompressConvolutionRewriter>(ctx, _enableExplicitDistributedTensorAttr, log);
    patterns.add<VPU::NCEInterpolateRewriter>(ctx, _enableExplicitDistributedTensorAttr, log);
    patterns.add<NCEPermuteRewriter>(ctx, _enableExplicitDistributedTensorAttr, log);
}
