//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPU/impl/make_ops_with_distributed_tensor_strategy.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/common_rewriters/make_ops_with_distributed_tensor.hpp"
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
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape(), strategy);
    auto distributedOutputTensorType = getDistributedOutputTensorType(
            clusteredOp, numClusters, strategy, outputTensorType, _enableExplicitDistributedTensorAttr);

    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles =
            getIntArrayAttr(ctx, getActivationTensorNumTiles(clusteredOp, numClusters.getInt(), strategy));

    mlir::ArrayAttr activationAlignmentAttr = nullptr;
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
    if (newOp->hasAttr("multiClusterStrategy")) {
        newOp->removeAttr("multiClusterStrategy");
    }
    auto outputCopyOp =
            rewriter.create<VPU::CopyOp>(newOp->getLoc(), origOp->getResult(0).getType(), newOp->getResult(0), nullptr);

    rewriter.replaceOp(origOp, outputCopyOp);

    return mlir::success();
}

}  // namespace

//
// MakeOpsWithDistributedTensorStrategy
//

void VPU::arch40xx::MakeOpsWithDistributedTensorStrategy::addPatterns(mlir::RewritePatternSet& patterns,
                                                                      Logger& log) const {
    auto ctx = patterns.getContext();
    patterns.add<VPU::NCEConvolutionRewriter>(ctx, _overlapParamsLookup, _enableExplicitDistributedTensorAttr, log);
    patterns.add<VPU::NCEDepthConvolutionRewriter>(ctx, _overlapParamsLookup, _enableExplicitDistributedTensorAttr,
                                                   log);
    patterns.add<VPU::NCEMaxPoolRewriter>(ctx, _overlapParamsLookup, _enableExplicitDistributedTensorAttr, log);
    patterns.add<VPU::NCEAveragePoolRewriter>(ctx, _overlapParamsLookup, _enableExplicitDistributedTensorAttr, log);
    patterns.add<VPU::NCEEltwiseRewriter>(ctx, _overlapParamsLookup, _enableExplicitDistributedTensorAttr, log);
    patterns.add<VPU::NCESWRewriter>(ctx, _overlapParamsLookup, _enableExplicitDistributedTensorAttr, log);
    patterns.add<VPU::NCECompressConvolutionRewriter>(ctx, _overlapParamsLookup, _enableExplicitDistributedTensorAttr,
                                                      log);
    patterns.add<VPU::NCEInterpolateRewriter>(ctx, _overlapParamsLookup, _enableExplicitDistributedTensorAttr, log);
    patterns.add<VPU::NCEMatMulRewriter>(ctx, _overlapParamsLookup, _enableExplicitDistributedTensorAttr, log);
    patterns.add<NCEPermuteRewriter>(ctx, _enableExplicitDistributedTensorAttr, log);
}
