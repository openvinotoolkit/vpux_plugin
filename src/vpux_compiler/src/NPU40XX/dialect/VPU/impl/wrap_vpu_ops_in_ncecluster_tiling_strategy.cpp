//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPU/impl/wrap_vpu_ops_in_ncecluster_tiling_strategy.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/common_rewriters/wrap_vpu_ops_in_ncecluster_tiling.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"

using namespace vpux;

namespace {

//
// NCEPermuteRewriter
//
// For NPU40XX logic is simpler than NPU37XX because all the OVERLAP
// needed for the next operation can be produced strictly at ODU
// and there is no need for the complicated logic of fusing overlap params.
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

    const auto strategy = origOp.getMultiClusterStrategy().value();
    auto outputTensorType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape()[Dims4D::Act::C], strategy);
    auto distributedOutputTensorType = getDistributedOutputTensorType(
            clusteredOp, numClusters, strategy, outputTensorType, _enableExplicitDistributedTensorAttr);

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);
    if (activationAlignment.has_value()) {
        activationAlignmentAttr = getIntArrayAttr(ctx, activationAlignment.value());
    }

    const auto distributedActivationCopyOp = createDistributedCopyIn(
            clusteredOp, origOp.getInput(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignmentAttr, strategy, _enableExplicitDistributedTensorAttr);

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
            origOp->getLoc(), distributedOutputTensorType, mlir::ValueRange{distributedActivationCopyOp->getResult(0)},
            bodyBuilder);

    const auto outputCopyOp = createDistributedCopyOut(origOp, clusterTilingOp);
    const auto origOutput = origOp->getResult(0);
    origOutput.replaceAllUsesWith(outputCopyOp->getResult(0));
    rewriter.replaceOp(origOp, outputCopyOp->getResult(0));

    return mlir::success();
}

}  // namespace

//
// WrapVPUOpsInNCEClusterTilingStrategy
//

void VPU::arch40xx::WrapVPUOpsInNCEClusterTilingStrategy::addPatterns(mlir::RewritePatternSet& patterns,
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
