//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/overlap_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sw_utils.hpp"

#include <llvm/ADT/TypeSwitch.h>

using namespace vpux;
using namespace VPU;

namespace {

SmallVector<int64_t> getOutAlignment(VPU::ClusteredOpInterface clusteredOp, int64_t numClusters,
                                     VPU::MultiClusterStrategy customStrategy,
                                     ArrayRef<vpux::NDTypeInterface> inputTypes, vpux::NDTypeInterface outputType) {
    const auto outputTensorNumTiles =
            vpux::VPU::getOutputTensorNumTiles(clusteredOp, numClusters, customStrategy, outputType);
    SmallVector<int64_t> outputAlignmentArr = {};
    // Set output alignment for HW layer
    auto clusteredHWNeedsChannelAlignment = [](const VPU::ClusteredOpInterface& hwOp) -> bool {
        return mlir::isa<VPU::NCEOpInterface>(*hwOp);
    };
    auto inputAlignment = vpux::VPU::getActivationTensorAlignment(clusteredOp, numClusters, customStrategy);
    if (mlir::isa<VPU::NCEEltwiseOp, VPU::NCEPermuteOp, VPU::ConcatOp>(clusteredOp.getOperation()) &&
        inputAlignment.has_value()) {
        // Eltwise input and output must have the same alignment/shape due to the hardware limitation
        outputAlignmentArr = inputAlignment.value();
    } else if (clusteredHWNeedsChannelAlignment(clusteredOp)) {
        const auto outputAlignment = getOutputTensorAlignment(customStrategy);
        if (outputAlignment.has_value()) {
            outputAlignmentArr = outputAlignment.value();
        }
    }

    // Set output alignment for SW layer
    if (mlir::isa<VPU::SWOpInterface>(clusteredOp.getOperation())) {
        if (inputTypes.empty()) {
            // If any input has alignment, output should have alignment as well
            for (auto operand : clusteredOp.getOperation()->getOperands()) {
                auto operandType = operand.getType().cast<NDTypeInterface>();
                if (isSWOpWithAlignedOutputChannelReq(clusteredOp, operandType, outputType)) {
                    VPUX_THROW_UNLESS(isSWOpWithAlignedInputChannelReq(clusteredOp, operandType, outputType),
                                      "SwOp input should have alignment at '{0}'", clusteredOp->getLoc());
                    outputAlignmentArr = DISTRIBUTED_C_ALIGNMENT;
                    break;
                }
            }
        } else {
            for (auto inputType : inputTypes) {
                if (isSWOpWithAlignedOutputChannelReq(clusteredOp, inputType, outputType)) {
                    VPUX_THROW_UNLESS(isSWOpWithAlignedInputChannelReq(clusteredOp, inputType, outputType),
                                      "SwOp input should have alignment at '{0}'", clusteredOp->getLoc());
                    outputAlignmentArr = DISTRIBUTED_C_ALIGNMENT;
                }
            }
        }
    }

    // Set output alignment for DepthToSpace, Width and Height must be aligned to block size
    if (auto depthToSpaceOp = mlir::dyn_cast<VPU::DepthToSpaceOp>(clusteredOp.getOperation())) {
        auto blockSize = depthToSpaceOp.getBlockSize();

        VPUX_THROW_WHEN(outputTensorNumTiles.size() != 4, "Expected 4D outputTensorNumTiles, but got {0} dimensions",
                        outputTensorNumTiles.size());
        SmallVector<int64_t> DISTRIBUTED_D2S_ALIGNMENT(outputTensorNumTiles.size(), 1);

        for (size_t i = 0; i < outputTensorNumTiles.size(); ++i) {
            if (outputTensorNumTiles[i] > 1) {
                int64_t tileIndex = checked_cast<int64_t>(i);
                if (tileIndex == Dims4D::Act::W.ind() || tileIndex == Dims4D::Act::H.ind()) {
                    DISTRIBUTED_D2S_ALIGNMENT[tileIndex] = blockSize;
                }
            }
        }

        outputAlignmentArr = std::move(DISTRIBUTED_D2S_ALIGNMENT);
    }
    return outputAlignmentArr;
}

bool hasOutputSpillingWithDuplicatedMode(mlir::Operation* op) {
    for (auto userOp : op->getUsers()) {
        // Skip cast ops
        while (auto userCastOp = mlir::dyn_cast_or_null<VPU::DistributedCastOpInterface>(userOp)) {
            if (hasMultiBranches(userOp)) {
                break;
            }
            userOp = *(userOp->getUsers().begin());
        }

        // If user op is not clustered op, there will definitely be output spilling
        if (!mlir::isa_and_nonnull<VPU::ClusteredOpInterface>(userOp)) {
            return true;
        }

        auto userClusteredOp = mlir::cast<VPU::ClusteredOpInterface>(userOp);
        auto userMCStrategy = userClusteredOp.getMultiClusterStrategy();
        // We consider there is spilling when user dosen't have mc strategy
        if (!userMCStrategy.has_value()) {
            return true;
        }

        if (userMCStrategy.value() == VPU::MultiClusterStrategy::Clustering) {
            // No spilling if user strategy is Clustering
            continue;
        }

        if ((userMCStrategy.value() == VPU::MultiClusterStrategy::SplitOverHeight) ||
            (userMCStrategy.value() == VPU::MultiClusterStrategy::SplitOverHeightOverlapped) ||
            (userMCStrategy.value() == VPU::MultiClusterStrategy::HKSwitch)) {
            // Currently, DUP->SEG(over H) has no spilling by adjusting workload offsets for NCEOps only
            // SWOp support is tracted by: E#118242
            if (!mlir::isa<VPU::NCEOpInterface>(userOp)) {
                return true;
            }
            continue;
        }

        if (userMCStrategy.value() == VPU::MultiClusterStrategy::SplitOverKernel) {
            // If user is SWOp or NCEPermute with SOK strategy, there is spilling for
            //     CurrentOp(DUP) -> (SOC)SWOp/NCEPermuteOp
            // For other ops, there can be no spilling because they can have DUPLICATED input
            if (mlir::isa<VPU::SWOpInterface, VPU::NCEPermuteOp>(userOp)) {
                return true;
            }
            continue;
        }

        // There is spilling for other strategies
        return true;
    }

    // If there is no userOp, or all userOps passed the checks above,
    // we consider there is no output spilling.
    return false;
}

};  // namespace

// Update or remove alignment for Slice like ops when alignment on slice axis
// for cases like SOC MVN with shave tiling over C
/* #Case for update:
                DistributedBuffer {C=64, T=4, alignment=[1, 16, 1, 1]}
                                  |
               /                                      \
           Subview1                                Subview2
 {C=32, T=4, alignment=[1, 8, 1, 1]}      {C=32, T=4, alignment=[1, 8, 1, 1]}
              |                                        |
           MVN_SHAVE1                               MVN_SHAVE2
*/
/* #Case for remove:
                DistributedBuffer {1x48x88x128, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters =
   2 : i64, alignment = [1, 2, 1, 1], memory_shapes = [[1, 48, 88, 128], [1, 48, 88, 128]], ...}}
                                  |
                                  |
                        Subview {offset=[0, 0, 0, 0], shape=[1, 3, 88, 128]}
                                  |
                                  |
                DistributedBuffer {1x3x88x128, {mode = "DUPLICATED|SEGMENTED", num_tiles = [1, 2, 1, 1], num_clusters =
   2 : i64, memory_shapes = [[1, 3, 88, 128], [1, 3, 88, 128]], ...}
*/
VPU::DistributionInfoAttr vpux::VPU::updateSliceLikeOpsAlignment(mlir::MLIRContext* ctx, vpux::ShapeRef inShape,
                                                                 vpux::ShapeRef sliceShape,
                                                                 VPU::DistributionInfoAttr originDistribution) {
    if (originDistribution == nullptr || originDistribution.getAlignment() == nullptr) {
        return originDistribution;
    }

    const auto alignmentVec = parseIntArrayAttr<int64_t>(originDistribution.getAlignment());
    auto it = std::find_if(alignmentVec.begin(), alignmentVec.end(), [](auto val) {
        return val > 1;
    });
    if (it == alignmentVec.end()) {
        return originDistribution;
    }
    auto idx = std::distance(alignmentVec.begin(), it);
    const auto dimAlign = Dim(idx);

    // Alignment is not on slice axis, using original one
    if (inShape[dimAlign] == sliceShape[dimAlign]) {
        return originDistribution;
    }

    // Set proper alignment or discard it
    const auto getDistribution = [&](mlir::ArrayAttr alignment) -> VPU::DistributionInfoAttr {
        return VPU::DistributionInfoAttr::get(
                ctx, originDistribution.getMode(), originDistribution.getNumTiles(), originDistribution.getKernel(),
                originDistribution.getPads(), originDistribution.getStrides(), originDistribution.getNumClusters(),
                alignment, originDistribution.getUniformDistributedSegments(), originDistribution.getComputeShapes(),
                originDistribution.getComputeOffsets(), originDistribution.getMemoryShapes(),
                originDistribution.getMemoryOffsets(), originDistribution.getEqualMemoryAndComputeView());
    };

    if (inShape[dimAlign] % sliceShape[dimAlign]) {
        return getDistribution(nullptr);
    }

    // scaleFactor to control how to scale alignment according to sliceShape
    const auto scaleFactor = inShape[dimAlign] / sliceShape[dimAlign];
    if (alignmentVec[dimAlign.ind()] % scaleFactor) {
        return getDistribution(nullptr);
    }

    // Update alignment to be ( OrigAlignment / scaleFactor ) for each slice op
    const auto perSliceAlign = alignmentVec[dimAlign.ind()] / scaleFactor;
    llvm::SmallVector<int64_t> newAlignmentVec = {1, 1, 1, 1};
    newAlignmentVec[idx] = perSliceAlign;
    auto newAlignment = getIntArrayAttr(ctx, ArrayRef(newAlignmentVec));
    return getDistribution(newAlignment);
}

//
// Distributed tensor utilities
//

bool vpux::VPU::isSOCSegmentedOp(mlir::Operation* op) {
    if (auto vfOp = mlir::dyn_cast_or_null<VPU::VerticalFusionOp>(op)) {
        auto* lastOp = vfOp.getBody()->getTerminator()->getOperand(0).getDefiningOp();
        return isSOCSegmentedOp(lastOp);
    }
    return isSOCSegmentedSWOp(op) || isSOCSegmentedNCEOp(op);
}

bool vpux::VPU::isSOCSegmentedSWOp(mlir::Operation* op) {
    auto clusteredOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(op);
    if (clusteredOp == nullptr) {
        return false;
    }
    if (!mlir::isa<VPU::SWOpInterface>(op)) {
        return false;
    }

    // Here assuming SWop always can be SOC Segmented if SOK strategy is compatible (like in strategy greedy assignment
    // phase). E.g., CONV SOK -> MVN (unassigned), we should select SEGMENTED mode for CONV output to avoid not fit CMX
    auto strategy = clusteredOp.getMultiClusterStrategy();
    if (!strategy.has_value()) {
        const auto numTiles = VPU::getNumTiles(op);
        return clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverKernel, numTiles);
    }

    return strategy.value() == VPU::MultiClusterStrategy::SplitOverKernel;
}

bool vpux::VPU::isSOCSegmentedNCEOp(mlir::Operation* op) {
    auto clusteredOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(op);
    if (clusteredOp == nullptr) {
        return false;
    }
    if (!mlir::isa<VPU::NCEOpInterface>(op)) {
        return false;
    }
    auto strategy = clusteredOp.getMultiClusterStrategy();
    if (!strategy.has_value() || strategy.value() != VPU::MultiClusterStrategy::SplitOverKernel) {
        return false;
    }
    // Currently only assign SOC to NCEOps when their parents are definitely SOC. However, for some cases like
    // DWConv, SOC DPU calculation can be more performant, need to figure out how to balance this cost.
    // E#119992 to track this.
    auto parentOp = op->getOperand(0).getDefiningOp();
    if (auto vfOp = op->getParentOfType<VPU::VerticalFusionOp>()) {
        if (auto vfArg = op->getOperand(0).dyn_cast<mlir::BlockArgument>()) {
            parentOp = vfOp.getOperand(vfArg.getArgNumber()).getDefiningOp();
        }
    }
    return mlir::isa<VPU::NCEPermuteOp>(op) ||
           (mlir::isa<VPU::NCEDepthConvolutionOp, VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp>(op) &&
            isSOCSegmentedOp(parentOp));
}

bool vpux::VPU::inputProducersCompatible(mlir::Operation* op, mlir::DenseSet<mlir::Operation*> handledUsers) {
    // propagate tiled ops
    if (mlir::isa<VPU::ConcatOp>(op)) {
        return isSegmentedInputCompatible(op, std::move(handledUsers));
    }
    // propagate through slice
    if (mlir::isa<VPU::SliceOp>(op)) {
        return isSegmentedInputCompatible(op, std::move(handledUsers));
    }
    // propagate copy
    if (mlir::isa<VPU::CopyOp>(op)) {
        return isSegmentedInputCompatible(op, std::move(handledUsers));
    }

    if (auto clusterOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(op)) {
        // propagate copy
        auto innerCopy = clusterOp.getInnerTaskOpOfType<VPU::CopyOp>();
        if (innerCopy != nullptr) {
            return isSegmentedInputCompatible(op, std::move(handledUsers));
        }

        const auto outputs = clusterOp->getResults();
        VPUX_THROW_UNLESS(outputs.size() == 1, "Wrong outputs size: {0}", outputs.size());

        const auto output = *outputs.begin();

        auto getDistributedTensor = [](const mlir::Value value) -> VPU::DistributedTensorType {
            if (auto sparseTensor = value.getType().dyn_cast<VPU::SparseTensorType>()) {
                return sparseTensor.getData().dyn_cast<VPU::DistributedTensorType>();
            }
            return value.getType().dyn_cast<VPU::DistributedTensorType>();
        };

        auto distributedOutputType = getDistributedTensor(output);
        VPUX_THROW_WHEN(distributedOutputType == nullptr, "Wrong output type {0} for NCEClusterTilingOp {1}",
                        output.getType(), clusterOp);

        return VPU::isSegmentedOverC(distributedOutputType.getDistribution());
    }

    return isSOCSegmentedOp(op);
}

bool vpux::VPU::isSegmentedInputCompatible(mlir::Operation* op, mlir::DenseSet<mlir::Operation*> handledUsers) {
    // For SW kernel, SplitOverKernel means input is tiled on channel axis
    if (mlir::isa<VPU::SWOpInterface>(op)) {
        return true;
    }
    // For NCE.Permute, SplitOverKernel means input is tiled on channel axis
    if (mlir::isa<VPU::NCEPermuteOp>(op)) {
        return true;
    }
    if (mlir::isa<VPU::NCEConvolutionOp, VPU::NCECompressConvolutionOp>(op)) {
        // full input required
        return false;
    }

    if (auto vfOp = op->getParentOfType<VPU::VerticalFusionOp>()) {
        return isSegmentedInputCompatible(vfOp, handledUsers);
    }

    // ConcatOp may have multiple inputs
    for (auto input : op->getOperands()) {
        if (auto vfOp = input.getDefiningOp<VPU::VerticalFusionOp>()) {
            auto* lastVFOp = vfOp.getBody()->getTerminator()->getOperand(0).getDefiningOp();
            if (!inputProducersCompatible(lastVFOp)) {
                return false;
            }
        } else if (auto definingOp = input.getDefiningOp()) {
            if (!inputProducersCompatible(definingOp)) {
                return false;
            }
        }
        // check siblings
        handledUsers.insert(op);
        for (auto* user : input.getUsers()) {
            if (handledUsers.contains(user)) {
                continue;
            }

            // Workaround to avoid getting stuck in an infinite loop when
            // having complex Slice-Concat patterns.
            // TODO: Remove it once before & after tiling distribution
            // inconsistencies are solved by E#76321
            if (mlir::isa<VPU::SliceOp>(op) && mlir::isa<VPU::SliceOp, VPU::ConcatOp>(user)) {
                handledUsers.insert(user);
                continue;
            }

            // If at least one producer is not SEGMENTED SW as compute, broadcast the data
            if (!inputProducersCompatible(user, handledUsers)) {
                return false;
            }
            handledUsers.insert(user);
        }

        // Except Concat, we only take into account operand 0 for the op with multiple inputs.
        // e.g  VPU.NCE.DepthConvolution.
        if (!mlir::isa<VPU::ConcatOp>(op)) {
            break;
        }
    }
    return true;
}

bool isOutputConsumersCompatible(mlir::Operation* op) {
    auto allUsers = op->getResult(0).getUsers();
    if (allUsers.empty()) {
        return false;
    }
    auto maybeYieldConsumer = *(allUsers.begin());
    if (auto yieldCons = llvm::dyn_cast<VPU::YieldOp>(maybeYieldConsumer)) {
        if (auto vfOp = op->getParentOfType<VPU::VerticalFusionOp>()) {
            return isOutputConsumersCompatible(vfOp);
        }
    }

    for (auto* user : allUsers) {
        // TODO: The propagation for ConcatOp/SliceOp/VerticalFusion can be removed after E#76321 solved
        if (mlir::isa<VPU::ConcatOp>(user)) {
            return isOutputConsumersCompatible(user);
        }
        if (mlir::isa<VPU::SliceOp>(user)) {
            return isOutputConsumersCompatible(user);
        }

        // If at least one consumer is not SEGMENTED SW as compute, broadcast the data
        if (auto vfOp = llvm::dyn_cast<VPU::VerticalFusionOp>(user)) {
            auto vfOperands = vfOp.getOperands();
            auto operandFromOrigOp = llvm::find_if(vfOperands, [&](mlir::Value operand) {
                return operand == op->getResult(0);
            });

            VPUX_THROW_WHEN(operandFromOrigOp == vfOperands.end(),
                            "Cannot find operand of VerticalFusion op matching the result of predecessor op");

            const auto operandNum = std::distance(vfOperands.begin(), operandFromOrigOp);
            auto innerInput = vfOp.getBody()->getArguments()[operandNum];
            for (auto inputUser : innerInput.getUsers()) {
                if (!isSOCSegmentedSWOp(inputUser)) {
                    return false;
                }
            }
        } else if (!isSOCSegmentedSWOp(user)) {
            return false;
        }
    }
    return true;
}

bool vpux::VPU::isSOKSegmentedOutputCompatible(mlir::Operation* op) {
    // For SW kernel, SplitOverKernel means input is tiled on channel axis
    if (mlir::isa<VPU::SWOpInterface>(op)) {
        return true;
    }
    // For NCE.Permute, SplitOverKernel means input is tiled on channel axis
    if (mlir::isa<VPU::NCEPermuteOp>(op)) {
        return true;
    }

    // force SEG -> DWConv -> SEG or SEG|DUP -> DWConv -> SEG|DUP to avoid accuracy issue
    if (mlir::isa<VPU::NCEDepthConvolutionOp, NCEMaxPoolOp, NCEAveragePoolOp>(op)) {
        if (VPU::getArch(op) == VPU::ArchKind::NPU40XX) {
            auto dstOrder = op->getResult(0).getType().cast<NDTypeInterface>().getDimsOrder();
            // Here we have two cases to choose SOC as default for Depthwise ops:
            //   1. The output consumer is compatible with SEGMENTED mode
            //   2. There is spilling with DUPLICATED mode output and the ouput order is NCXX.
            //       - In this case, SOH is most likely to be choosen because DUPLICATED mode is likely
            //         to cause tiling, but if we choose SOC, the performance can be better because Depthwise
            //         ops need workload splits on channel, so that SOC mode can produce fewer workload splits.
            //       - The output order being limited to NCXX is because there is no stride DMA spilling in this
            //         case, otherwise the spilling costs more than the compute improvement.
            return isOutputConsumersCompatible(op) || (hasOutputSpillingWithDuplicatedMode(op) &&
                                                       (dstOrder == DimsOrder::NCHW || dstOrder == DimsOrder::NCWH));
        }

        return isSegmentedInputCompatible(op);
    }

    // force SEG -> DPU -> SEG prevent SEG -> DPU -> SEG|DUP
    // re-enable with RT support E#66658
    if (isSegmentedInputCompatible(op)) {
        return true;
    }

    // check consumers
    return isOutputConsumersCompatible(op);
}

// This method computes the number of clusters to be used for an individual SOK
// layer such that additional alignment of the per cluster output channels is not required.
// Example: For 80 output channel / 4 clusters = [20, 20, 20, 20] output channels per cluster.
// 20 is not aligned to 16. Therefore, the compiler should only execute this layer on 3 clusters.
// This would result in [32, 32, 16] output channels per cluster.
int64_t vpux::VPU::getNumberOfClustersForSOKToAvoidAlignment(int64_t outputChannels, int64_t numClustersToUseForLayer,
                                                             bool uniformDistributedSegments) {
    for (int64_t clusters = numClustersToUseForLayer; clusters >= 1; clusters--) {
        if (uniformDistributedSegments) {
            // For VPUX40XX there's no limitation on how the segments need to be equal to eachother.
            // A balanced segmentation is prefered for performance.
            // A depth of 96 is best split across 4 clusters as [32, 32, 16, 16]

            // Align downwards to the next mutiple of KMB_DPU_CHANNELS_ALIGNMENT
            auto baselineChannels = alignValDown<int64_t>(outputChannels / clusters, KMB_DPU_CHANNELS_ALIGNMENT);
            int64_t remainder = outputChannels - baselineChannels * clusters;
            // Even if baseline itself is > 0 favor the cases where remainder is a multiple of alignment
            if (baselineChannels > 0 && !(remainder % KMB_DPU_CHANNELS_ALIGNMENT)) {
                return clusters;
            }
        } else {
            // For VPUX3XXX architectures, there's unwritten contract that the first N-1 cluster segments
            // all need to be equal, and the last segment can be equal or smaller.
            auto alignedOutputChannels =
                    alignValUp<int64_t>(divUp(outputChannels, clusters), KMB_DPU_CHANNELS_ALIGNMENT);
            int64_t remainder = outputChannels - (clusters - 1) * alignedOutputChannels;
            if (remainder > 0) {
                return clusters;
            }
        }
    }
    return 1;
}

int64_t vpux::VPU::getNumberOfClustersForSpatialDim(int64_t outputSpatialDim, int64_t numClustersForCompilation,
                                                    bool uniformDistributedSegments) {
    for (int64_t clusters = numClustersForCompilation; clusters >= 1; clusters--) {
        if (uniformDistributedSegments) {
            // For VPUX40XX there's no limitation on how the segments need to be equal to eachother.
            // A balanced segmentation is prefered for performance.
            // A height of 6 is best split across 4 clusters as [2, 2, 1, 1]
            auto baselineHeight = outputSpatialDim / clusters;
            if (baselineHeight > 0) {
                return clusters;
            }
        } else {
            // For VPUX3XXX architectures, there's unwritten contract that the first N-1 cluster segments
            // all need to be equal, and the last segment can be equal or smaller.
            auto alignedOutputSpatialDim = divUp(outputSpatialDim, clusters);
            int64_t remainder = outputSpatialDim - (clusters - 1) * alignedOutputSpatialDim;
            if (remainder > 0) {
                return clusters;
            }
        }
    }
    return 1;
}

SmallVector<int64_t> vpux::VPU::getActivationTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                                            int64_t numClustersAvailableForCompilation,
                                                            VPU::MultiClusterStrategy strategy,
                                                            vpux::NDTypeInterface inputType) {
    auto inputTensorType =
            inputType != nullptr ? inputType : clusteredOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputTensorType.getShape();
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
        strategy == VPU::MultiClusterStrategy::SplitOverHeight || strategy == VPU::MultiClusterStrategy::HKSwitch) {
        return {1, 1, numClustersAvailableForCompilation, 1};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        if (isSegmentedInputCompatible(clusteredOp.getOperation())) {
            auto IC = inputShape[Dims4D::Act::C];
            int64_t numClustersToUseForLayer = std::min(numClustersAvailableForCompilation, IC);
            // E-143638: DequantizeOp is used to processed weights, not activation, a general
            // solution to distinguish weights and activations for distributed tensor utils
            if (mlir::isa<VPU::DequantizeOp>(clusteredOp.getOperation())) {
                auto OC = inputShape[Dims4D::Filter::OC];
                numClustersToUseForLayer = std::min(numClustersAvailableForCompilation, OC);
                return {numClustersToUseForLayer, 1, 1, 1};
            }
            return {1, numClustersToUseForLayer, 1, 1};
        }
        return {1, 1, 1, 1};
    } else if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return {1, 1, 1, 1};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverWidth) {
        return {1, 1, 1, numClustersAvailableForCompilation};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverBatch) {
        const auto batchTilingNum = getOptimalNumClusters(clusteredOp, inputShape, strategy);
        return {batchTilingNum, 1, 1, 1};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverGroup) {
        return {numClustersAvailableForCompilation, 1, 1, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "activation tensor",
                   strategy);
    }
}

bool vpux::VPU::isDWOpAndNeedsAlign(ArchKind arch, VPUIP::NCETaskType nceTaskType) {
    bool isDWOp = nceTaskType == VPUIP::NCETaskType::DWCONV || nceTaskType == VPUIP::NCETaskType::MAXPOOL ||
                  nceTaskType == VPUIP::NCETaskType::AVEPOOL;
    return (arch == VPU::ArchKind::NPU37XX) && isDWOp;
}

bool vpux::VPU::isEltwiseOpAndNeedsAlign(VPU::ClusteredOpInterface clusteredOp) {
    auto nceEltwiseOp = mlir::dyn_cast<VPU::NCEEltwiseOp>(clusteredOp.getOperation());
    if (nceEltwiseOp == nullptr) {
        return false;
    }

    // Find if there exists a non-eltwise nceOp with SOH in eltwise subgraph
    llvm::SmallPtrSet<mlir::Operation*, 16> processedInputOps;
    std::deque<mlir::Value> inputs = {nceEltwiseOp->getOperand(0), nceEltwiseOp->getOperand(1)};
    while (!inputs.empty()) {
        const auto currentInput = inputs.front();
        // skip processed input
        if (auto defOp = currentInput.getDefiningOp()) {
            if (processedInputOps.count(defOp) > 0) {
                inputs.pop_front();
                continue;
            }
        }
        for (auto userOp : currentInput.getUsers()) {
            // Skip non-clustered and non-NCE ops
            if (!mlir::isa<VPU::ClusteredOpInterface>(userOp) || !mlir::isa<VPU::NCEOpInterface>(userOp)) {
                continue;
            }

            // There are 2 scenarios that we need to set alignment attr to eltwises
            // Scenario 1:
            //   Has one sibling op with SOH whose input needs alignment
            //                 AnyOp      AnyOp
            //                 /   \       /
            //            ConvOp   *EltwiseOp
            // Scenario 2:
            //   Has one descendant op with SOH whose input needs alignment
            //               *EltwiseOp    AnyOp
            //                        \    /
            //                       EltwiseOp
            //                           |
            //                         ConvOp
            if (auto userEltwiseOp = mlir::dyn_cast<VPU::NCEEltwiseOp>(userOp)) {
                // Should also find in child eltwiseOp's siblings and children
                auto userEltwiseInput1 = userEltwiseOp.getInput1();
                if (userEltwiseInput1 != currentInput &&
                    processedInputOps.count(userEltwiseInput1.getDefiningOp()) == 0) {
                    inputs.push_back(userEltwiseInput1);
                }
                auto userEltwiseInput2 = userEltwiseOp.getInput2();
                if (userEltwiseInput2 != currentInput &&
                    processedInputOps.count(userEltwiseInput2.getDefiningOp()) == 0) {
                    inputs.push_back(userEltwiseInput2);
                }
                auto userEltwiseOutput = userEltwiseOp.getOutput();
                if (processedInputOps.count(userEltwiseOutput.getDefiningOp()) == 0) {
                    inputs.push_back(userEltwiseOutput);
                }
            } else {
                // Check if it's a non-eltwise with SOH
                auto userNceOp = mlir::cast<VPU::ClusteredOpInterface>(userOp);
                auto strategy = userNceOp.getMultiClusterStrategy();
                if (strategy.has_value() && (strategy.value() == VPU::MultiClusterStrategy::SplitOverHeight ||
                                             strategy.value() == VPU::MultiClusterStrategy::HKSwitch)) {
                    return true;
                }
            }
        }
        processedInputOps.insert(currentInput.getDefiningOp());
        inputs.pop_front();
    }
    return false;
}

bool vpux::VPU::isSWOpChannelAlignmentCompatible(VPU::ClusteredOpInterface swOp, vpux::NDTypeInterface inputType,
                                                 vpux::NDTypeInterface outputType) {
    if (!mlir::isa<VPU::SWOpInterface>(swOp.getOperation())) {
        return false;
    }

    if (swOp->getOperands().size() != 1 && swOp->getResults().size() != 1) {
        return false;
    }

    const auto strategy = swOp.getMultiClusterStrategy();
    if (!strategy.has_value()) {
        return false;
    }

    // Only when SW Op with Clustering and SOK strategy
    auto actInputC = inputType.getShape()[Dims4D::Act::C];
    auto actOutputC = outputType.getShape()[Dims4D::Act::C];
    auto alignment = DISTRIBUTED_C_ALIGNMENT[Dims4D::Act::C.ind()];

    if (strategy.value() == VPU::MultiClusterStrategy::Clustering) {
        return (actInputC % alignment == 0) && (actOutputC % alignment == 0);
    } else if (strategy.value() == VPU::MultiClusterStrategy::SplitOverKernel) {
        auto module = swOp->getParentOfType<mlir::ModuleOp>();
        auto tileCount = IE::getTileExecutor(module).getCount();
        if (actInputC % (alignment * tileCount) == 0 && actOutputC % (alignment * tileCount) == 0) {
            // Input and output can be divided evenly into each tile
            return true;
        }
        if (swOp->hasTrait<VPU::EltwiseOp>()) {
            // if Input and output are divided unevenly, need to check the segmented shape can be created or not. It's
            // not supported by non-eltwise op.
            // For example, if input [1,96, 1, 1] with 16 alignment on channel, and output shape is [1, 48, 1, 1].
            // If input is segmented into 2 tiles, then
            // input tiles : [1, 48, 1, 1]
            //               [1, 48, 1, 1].
            // If alignment is added to the output, the output shape :
            // output tiles : [1, 32, 1, 1]
            //                [1, 16, 1, 1]
            SmallVector<int64_t> alignmentArray = {1, alignment, 1, 1};
            SmallVector<int64_t> tilingScheme = {0, tileCount, 0, 0};
            auto uniformDistributedSegments = VPU::isUniformDistributedSegmentsSupported(swOp);
            auto inputSegmentedShape =
                    VPU::splitSegmentedShape(to_small_vector(inputType.getShape()), tilingScheme, tileCount,
                                             Dims4D::Act::C.ind(), alignmentArray, uniformDistributedSegments);
            if (!inputSegmentedShape.has_value()) {
                return false;
            }
            auto segmentedShapes = inputSegmentedShape.value();
            VPUX_THROW_WHEN(segmentedShapes.empty(), "Segmented shape list is empty");
            return segmentedShapes.back()[Dims4D::Act::C] % alignment == 0;
        }
    }

    return false;
}

bool isHSegmentedType(vpux::VPU::DistributedTensorType distributedType) {
    auto mode = distributedType.getDistribution().getMode().getValue();
    if (mode == VPU::DistributionMode::OVERLAPPED) {
        // SplitOverHOverlapped
        return true;
    }
    if (mode != VPU::DistributionMode::SEGMENTED) {
        // Clustering or SplitOverKernel
        return false;
    }
    auto numTilesAttr = distributedType.getDistribution().getNumTiles();
    if (numTilesAttr == nullptr) {
        return false;
    }
    auto numTiles = parseIntArrayAttr<int64_t>(numTilesAttr);
    return numTiles[Dims4D::Act::H.ind()] > 1;
}

bool isSWParentAlignmentAtChannel(VPU::ClusteredOpInterface swOp) {
    const auto curStrategy = swOp.getMultiClusterStrategy();
    if (!curStrategy.has_value()) {
        return false;
    }

    auto isParentAlignmentAtChannel = [&](mlir::Value input) -> bool {
        const auto inputType = input.getType().cast<NDTypeInterface>();
        const auto mode = getSWInputTensorDistributionMode(swOp, curStrategy.value(), inputType);
        if (mode != DistributionMode::SEGMENTED) {
            return false;
        }

        auto isClusteredCopy = [](mlir::Operation* op) -> bool {
            if (auto clusteredOp = mlir::dyn_cast<VPU::NCEClusterTilingOp>(op)) {
                auto innerCopy = clusteredOp.getInnerTaskOpOfType<VPU::CopyOp>();
                if (innerCopy == nullptr) {
                    return false;
                }
                return true;
            }
            return false;
        };

        auto parentOp = input.getDefiningOp();
        while (parentOp != nullptr && (mlir::isa<VPU::ViewLikeOpInterface>(parentOp) || isClusteredCopy(parentOp))) {
            parentOp = parentOp->getOperand(0).getDefiningOp();
        }
        if (parentOp == nullptr) {
            return false;
        }

        if (mlir::isa<VPU::NCEOpInterface>(parentOp)) {
            auto clusteredNCEOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(parentOp);
            if (clusteredNCEOp == nullptr) {
                return false;
            }

            auto strategy = clusteredNCEOp.getMultiClusterStrategy();
            if (!strategy.has_value()) {
                return false;
            }

            const auto mcStrategy = strategy.value();
            const bool isSplitOnWidthOrHeight = mcStrategy == VPU::MultiClusterStrategy::SplitOverWidth ||
                                                mcStrategy == VPU::MultiClusterStrategy::SplitOverHeight ||
                                                mcStrategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped;
            return !isSplitOnWidthOrHeight;
        } else if (mlir::isa<VPU::SWOpInterface>(parentOp)) {
            auto clusteredSwOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(parentOp);
            if (clusteredSwOp == nullptr) {
                return false;
            }
            auto swInType = parentOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
            auto swOutType = parentOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
            if (!isSWOpChannelAlignmentCompatible(clusteredSwOp, swInType, swOutType)) {
                return false;
            }
            return isSWParentAlignmentAtChannel(clusteredSwOp);
        }

        if (!mlir::isa<VPU::NCEClusterTilingOp>(parentOp)) {
            return false;
        }

        auto parentDistributedType = parentOp->getResult(0).getType().cast<vpux::VPU::DistributedTensorType>();
        if (isHSegmentedType(parentDistributedType)) {
            // SOH parent cannot be compatible with Clustering/SOK SW op
            return false;
        }
        auto parentAlignment = parentDistributedType.getDistribution().getAlignment();
        return parentAlignment != nullptr;
    };

    return llvm::any_of(swOp->getOperands(), isParentAlignmentAtChannel);
}

bool isSWUsersAlignmentAtChannel(VPU::ClusteredOpInterface swOp) {
    for (auto childOp : swOp->getResult(0).getUsers()) {
        while (childOp != nullptr && mlir::isa<VPU::ViewLikeOpInterface>(childOp)) {
            childOp = *childOp->getResult(0).getUsers().begin();
            if (hasMultiBranches(childOp)) {
                return false;
            }
        }
        if (childOp == nullptr || !mlir::isa<VPU::NCEOpInterface>(childOp) ||
            !mlir::isa<VPU::ClusteredOpInterface>(childOp)) {
            return false;
        }

        auto clusteredNCEOp = mlir::cast<VPU::ClusteredOpInterface>(childOp);
        auto strategy = clusteredNCEOp.getMultiClusterStrategy();
        // Only add alignment when the child strategy is not split on width or height, to keep subgraph consistent
        if (strategy.has_value()) {
            auto mcStrategy = strategy.value();
            bool isSplitOnWidthOrHeight = mcStrategy == VPU::MultiClusterStrategy::SplitOverWidth ||
                                          mcStrategy == VPU::MultiClusterStrategy::SplitOverHeight ||
                                          mcStrategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped;
            if (!isSplitOnWidthOrHeight) {
                return true;
            }
        }
    }
    return false;
}

// Adjust inputType alignment for SW op to avoid spilling.
// For example:
//  - Conv (SOK) -> SW (SOK), the input of SW can set alignment of channel to 16
bool vpux::VPU::isSWOpWithAlignedInputChannelReq(VPU::ClusteredOpInterface swOp, vpux::NDTypeInterface inputType,
                                                 vpux::NDTypeInterface outputType) {
    auto swInType = inputType != nullptr ? inputType : swOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    auto swOutType = outputType != nullptr ? outputType : swOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    if (isSWOpChannelAlignmentCompatible(swOp, swInType, swOutType)) {
        return isSWUsersAlignmentAtChannel(swOp) || isSWParentAlignmentAtChannel(swOp);
    }
    return false;
}

// Adjust inputType alignment for SW op to avoid spilling.
// For example:
//  - SW (Clustering) -> Conv (SOK), the output of SW can set alignment of channel to 16
bool vpux::VPU::isSWOpWithAlignedOutputChannelReq(VPU::ClusteredOpInterface swOp, vpux::NDTypeInterface inputType,
                                                  vpux::NDTypeInterface outputType) {
    auto swInType = inputType != nullptr ? inputType : swOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    auto swOutType = outputType != nullptr ? outputType : swOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    if (isSWOpChannelAlignmentCompatible(swOp, swInType, swOutType)) {
        return isSWUsersAlignmentAtChannel(swOp) || isSWParentAlignmentAtChannel(swOp);
    }
    return false;
}

std::optional<SmallVector<int64_t>> vpux::VPU::getActivationTensorAlignment(VPU::ClusteredOpInterface clusteredOp,
                                                                            int64_t numClusters,
                                                                            VPU::MultiClusterStrategy strategy,
                                                                            vpux::NDTypeInterface inputType,
                                                                            vpux::NDTypeInterface outputType) {
    if (mlir::isa<VPU::SWOpInterface>(clusteredOp.getOperation())) {
        if (isSWOpWithAlignedInputChannelReq(clusteredOp, inputType, outputType)) {
            return DISTRIBUTED_C_ALIGNMENT;
        }
        return std::nullopt;
    }

    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::Clustering) {
        if (!mlir::isa<VPU::ConcatOp>(clusteredOp)) {
            return DISTRIBUTED_C_ALIGNMENT;
        }

        const auto outputShape = outputType != nullptr
                                         ? outputType.getShape()
                                         : clusteredOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape();
        auto uniformDistributedSegments = VPU::isUniformDistributedSegmentsSupported(clusteredOp.getOperation());
        const auto numClustersToUseForLayer = getNumberOfClustersForSOKToAvoidAlignment(
                outputShape[Dims4D::Act::C], numClusters, uniformDistributedSegments);

        if (numClustersToUseForLayer == 1) {
            return std::nullopt;
        }

        return DISTRIBUTED_C_ALIGNMENT;

    } else if (strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
               strategy == VPU::MultiClusterStrategy::HKSwitch) {
        auto operation = clusteredOp.getOperation();
        auto arch = getArch(operation);
        const std::set<VPU::ArchKind> incompatibleTargets = {
                VPU::ArchKind::NPU40XX,
        };
        if (incompatibleTargets.count(arch) > 0) {
            return std::nullopt;
        }

        if (mlir::isa<VPU::NCEConvolutionOp, VPU::NCEInterpolateOp>(operation) ||
            ((arch == VPU::ArchKind::NPU37XX) &&
             mlir::isa<VPU::NCEDepthConvolutionOp, VPU::NCEMaxPoolOp, VPU::NCEAveragePoolOp,
                       VPU::NCECompressConvolutionOp>(operation)) ||
            isEltwiseOpAndNeedsAlign(clusteredOp)) {
            if (inputType == nullptr) {
                inputType = clusteredOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
            }
            const auto inputShape = inputType.getShape();
            const auto isInputSparse = inputType.isa<VPU::SparseTensorType>();
            const auto heightAlignment = getSOHMinimalHeightAlignment(inputShape, numClusters, isInputSparse, arch);
            if (heightAlignment <= 1 || heightAlignment >= inputShape[Dims4D::Act::H]) {
                return std::nullopt;
            }

            return SmallVector<int64_t>{1, 1, heightAlignment, 1};
        }
    }
    return std::nullopt;
}

SmallVector<int64_t> vpux::VPU::getOutputTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                                        int64_t numClustersAvailableForCompilation,
                                                        VPU::MultiClusterStrategy strategy,
                                                        vpux::NDTypeInterface outputType) {
    const auto outputShape = outputType != nullptr
                                     ? outputType.getShape()
                                     : clusteredOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape();
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
        strategy == VPU::MultiClusterStrategy::SplitOverHeight || strategy == VPU::MultiClusterStrategy::HKSwitch) {
        return {1, 1, numClustersAvailableForCompilation, 1};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        const auto numClustersToUseForLayer = getOptimalNumClusters(clusteredOp, outputShape, strategy);
        // E-143638: DequantizeOp is used to processed weights, not activation, a general
        // solution to distinguish weights and activations for distributed tensor utils
        if (mlir::isa<VPU::DequantizeOp>(clusteredOp.getOperation())) {
            auto OC = outputShape[Dims4D::Filter::OC];
            if (OC == 1) {
                return {1, numClustersToUseForLayer, 1, 1};
            }
            return {numClustersToUseForLayer, 1, 1, 1};
        }
        return {1, numClustersToUseForLayer, 1, 1};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverWidth) {
        return {1, 1, 1, numClustersAvailableForCompilation};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverBatch) {
        const auto batchTilingNum = getOptimalNumClusters(clusteredOp, outputShape, strategy);
        return {batchTilingNum, 1, 1, 1};
    } else if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return {1, 1, 1, 1};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverGroup) {
        return {numClustersAvailableForCompilation, 1, 1, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "output tensor",
                   strategy);
    }
}

std::optional<SmallVector<int64_t>> vpux::VPU::getOutputTensorAlignment(VPU::MultiClusterStrategy strategy) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::Clustering) {
        return DISTRIBUTED_C_ALIGNMENT;
    }

    return std::nullopt;
}

std::optional<vpux::NDTypeInterface> vpux::VPU::adjustOutputAlignmentForSOH(VPU::ClusteredOpInterface clusteredOp,
                                                                            vpux::NDTypeInterface originalDistType) {
    if (clusteredOp->getResult(0).use_empty()) {
        return std::nullopt;
    }

    if (mlir::isa<VPU::SWOpInterface>(clusteredOp.getOperation())) {
        return std::nullopt;
    }

    auto originalDistTypeIf = originalDistType.dyn_cast<VPU::DistributedTypeInterface>();
    VPUX_THROW_UNLESS(originalDistTypeIf != nullptr, "Expected type to be distributed, got {0}", originalDistType);
    VPUX_THROW_UNLESS(originalDistTypeIf.containsDistributedTypes(), "Type does not contain distributed components");
    const auto distributedTypes = originalDistTypeIf.getDistributedTypes();

    const auto distributedDataType = distributedTypes.front().cast<VPU::DistributedTensorType>();

    auto updateAlignment = [&](VPU::ClusteredOpInterface consumerOp, bool skipCmxCheck,
                               vpux::NDTypeInterface inputType = nullptr) -> std::optional<NDTypeInterface> {
        auto getAlignedDistributedTensorType =
                [&clusteredOp](ArrayRef<int64_t> alignment,
                               VPU::DistributedTensorType distType) -> VPU::DistributedTensorType {
            const auto newAlignmentAttr = getIntArrayAttr(clusteredOp->getContext(), alignment);
            auto distributedAttr = distType.getDistribution();
            auto newDistributedAttr = VPU::DistributionInfoAttr::get(
                    clusteredOp->getContext(), distributedAttr.getMode(), distributedAttr.getNumTiles(),
                    distributedAttr.getKernel(), distributedAttr.getPads(), distributedAttr.getStrides(),
                    distributedAttr.getNumClusters(), newAlignmentAttr, distributedAttr.getUniformDistributedSegments(),
                    distributedAttr.getComputeShapes(), distributedAttr.getComputeOffsets(),
                    distributedAttr.getMemoryShapes(), distributedAttr.getMemoryOffsets(),
                    distributedAttr.getEqualMemoryAndComputeView());
            return VPU::DistributedTensorType::get(clusteredOp->getContext(), distType.getShape().raw(),
                                                   distType.getElementType(), distType.getOrder(),
                                                   distType.getMemSpace(), newDistributedAttr);
        };

        const auto newAlignment = getActivationTensorAlignment(
                consumerOp, distributedDataType.getDistribution().getNumClusters().getInt(),
                VPU::MultiClusterStrategy::SplitOverHeight, inputType);
        if (!newAlignment.has_value()) {
            return std::nullopt;
        }

        SmallVector<VPU::DistributedTensorType> newDistributedTypes;
        for (auto type : distributedTypes) {
            auto distType = type.cast<VPU::DistributedTensorType>();
            newDistributedTypes.push_back(getAlignedDistributedTensorType(newAlignment.value(), distType));
        }

        if (originalDistType.isa<VPU::SparseTensorType>()) {
            VPUX_THROW_UNLESS(newDistributedTypes.size() >= 1, "Expected at least 1 distributed type, got {0}",
                              newDistributedTypes.size());
            const auto newDataType = newDistributedTypes[0];
            const auto newSMType = (newDistributedTypes.size() > 1) ? newDistributedTypes[1] : nullptr;
            const auto newSEType = (newDistributedTypes.size() > 2) ? newDistributedTypes[2] : nullptr;
            const auto newSparseOutputType = VPU::SparseTensorType::get(newDataType, newSMType, newSEType);
            if (skipCmxCheck || clusteredOp.doesLayerChangeOutputAlignmentFitIntoCMX(
                                        VPU::MultiClusterStrategy::SplitOverHeight, newSparseOutputType)) {
                return newSparseOutputType.cast<vpux::NDTypeInterface>();
            }
        }

        if (newDistributedTypes.size() == 1) {
            if (skipCmxCheck || clusteredOp.doesLayerChangeOutputAlignmentFitIntoCMX(
                                        VPU::MultiClusterStrategy::SplitOverHeight, newDistributedTypes[0])) {
                return newDistributedTypes[0].cast<vpux::NDTypeInterface>();
            }
        }

        return std::nullopt;
    };

    // If the nceOp is eltwise, the output alignment should be the same as input.
    if (mlir::isa<VPU::NCEEltwiseOp>(clusteredOp)) {
        return updateAlignment(clusteredOp, /*skipCmxCheck=*/true);
    }

    // optimization SOH -> SOH alignment to remove spilling
    // For multi-users just random choose one NCEOp for optimize
    // TODO: choose the best NCEOp or find least common multiple of all user's alignment
    for (auto consumerOp : clusteredOp->getResult(0).getUsers()) {
        // If user is a concatOp whose output shape is the same as the
        // output shape of nceOp in both H & W, adjust output alignment
        // with input of concatOp's users to enable cmx concat.
        if (auto concatOp = mlir::dyn_cast<VPU::ConcatOp>(consumerOp)) {
            auto concatOutputShape = getShape(concatOp->getResult(0));
            auto isHWShapeSame = llvm::all_of(concatOp.getInputs(), [&](mlir::Value input) {
                auto concatInputShape = input.getType().cast<vpux::NDTypeInterface>().getShape();
                return concatInputShape[Dims4D::Act::H] == concatOutputShape[Dims4D::Act::H] &&
                       concatInputShape[Dims4D::Act::W] == concatOutputShape[Dims4D::Act::W];
            });
            if (isHWShapeSame) {
                consumerOp = *consumerOp->getResult(0).getUsers().begin();
            }
        }

        if (!mlir::isa<VPU::NCEOpInterface>(consumerOp)) {
            continue;
        }

        auto consumerClusterOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(consumerOp);
        auto consumerMultiClusterStrategyAttr = consumerClusterOp.getMultiClusterStrategy();
        if (!consumerMultiClusterStrategyAttr.has_value()) {
            continue;
        }

        const auto strategy = consumerMultiClusterStrategyAttr.value();
        if (strategy != VPU::MultiClusterStrategy::SplitOverHeight && strategy != VPU::MultiClusterStrategy::HKSwitch) {
            continue;
        }

        return updateAlignment(consumerClusterOp, /*skipCmxCheck=*/false);
    }
    return std::nullopt;
}

SmallVector<int64_t> vpux::VPU::getWeightsTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                                         vpux::NDTypeInterface tensorType,
                                                         int64_t numClustersAvailableForCompilation,
                                                         VPU::MultiClusterStrategy strategy) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
        strategy == VPU::MultiClusterStrategy::SplitOverHeight || strategy == VPU::MultiClusterStrategy::Clustering ||
        strategy == VPU::MultiClusterStrategy::HKSwitch || strategy == VPU::MultiClusterStrategy::SplitOverBatch) {
        return {1, 1, 1, 1};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        auto OC = tensorType.getShape()[Dims4D::Filter::OC];
        auto uniformDistributedSegments = VPU::isUniformDistributedSegmentsSupported(clusteredOp);
        int64_t numClustersToUseForLayer = getNumberOfClustersForSOKToAvoidAlignment(
                OC, numClustersAvailableForCompilation, uniformDistributedSegments);
        return {numClustersToUseForLayer, 1, 1, 1};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverGroup) {
        return {numClustersAvailableForCompilation, 1, 1, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "weights tensor",
                   strategy);
    }
}

std::optional<SmallVector<int64_t>> vpux::VPU::getWeightsTensorAlignment(VPU::MultiClusterStrategy strategy) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel || strategy == VPU::MultiClusterStrategy::Clustering) {
        return SmallVector<int64_t>{16, 1, 1, 1};
    }
    return std::nullopt;
}

SmallVector<int64_t> vpux::VPU::getWeightsTableTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                                              vpux::NDTypeInterface tensorType,
                                                              int64_t numClustersAvailableForCompilation,
                                                              VPU::MultiClusterStrategy strategy) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
        strategy == VPU::MultiClusterStrategy::SplitOverHeight || strategy == VPU::MultiClusterStrategy::Clustering ||
        strategy == VPU::MultiClusterStrategy::HKSwitch || strategy == VPU::MultiClusterStrategy::SplitOverBatch) {
        return {1, 1, 1, 1};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        auto OC = tensorType.getShape()[Dims4D::Act::C];
        auto uniformDistributedSegments = VPU::isUniformDistributedSegmentsSupported(clusteredOp);
        int64_t numClustersToUseForLayer = getNumberOfClustersForSOKToAvoidAlignment(
                OC, numClustersAvailableForCompilation, uniformDistributedSegments);
        return {numClustersToUseForLayer, 1, 1, 1};
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverGroup) {
        return {numClustersAvailableForCompilation, 1, 1, 1, 1};
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the number of tiles for the "
                   "weights table tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getActivationTensorDistributionMode(VPU::ClusteredOpInterface clusteredOp,
                                                                VPU::MultiClusterStrategy strategy) {
    // Judge if we can select DistributionMode::DUPLICATED mode for the activation of SOH-like strategies
    // Todo: consider SOW, refer to ticket E#117156
    auto isDuplicatedModeForSOHLikeStrategy = [&]() {
        if (strategy != VPU::MultiClusterStrategy::SplitOverHeightOverlapped &&
            strategy != VPU::MultiClusterStrategy::SplitOverHeight && strategy != VPU::MultiClusterStrategy::HKSwitch) {
            return false;
        }

        auto op = clusteredOp.getOperation();

        // Note: disable concat as it is a complex topic
        // As concatOp has a special cmx-concat pattern check, thus the spilling may still exist even to assign
        // DUPLICATED
        if (mlir::isa<VPU::ConcatOp>(op)) {
            return false;
        }

        // For NCECompressConvolutionOp, the activation (without expansion) must have a channel size of
        // VPU_COMPRESSED_INPUT_CHANNEL_NUM (4). Otherwise, duplicated inputs with workload offsets cannot correctly
        // access the activation data for each cluster, leading to accuracy issues
        if (auto compressConv = mlir::dyn_cast<VPU::NCECompressConvolutionOp>(op)) {
            auto origChannelVal = static_cast<int64_t>(std::log2(compressConv.getCmSpPattern() + 1));
            if (origChannelVal != VPU::NCEInvariant::VPU_COMPRESSED_INPUT_CHANNEL_NUM) {
                return false;
            }
        }

        // For sw ops, current solution is dependent on workload offsets adjust so not support sw ops
        // Todo: refer to ticket E#118242: use per cluster unrolling to solve it
        if (mlir::isa<VPU::SWOpInterface>(op)) {
            return false;
        }

        llvm::SmallVector<bool> eltwiseInputsCompatible = {false, false};
        for (auto operand : op->getOperands() | indexed) {
            auto producerOp = operand.value().getDefiningOp();
            // Skip cast ops
            while (mlir::isa_and_nonnull<VPU::DistributedCastOpInterface>(producerOp)) {
                if (hasMultiBranches(producerOp)) {
                    break;
                }
                producerOp = producerOp->getOperand(0).getDefiningOp();
            }
            if (producerOp == nullptr) {
                return false;
            }

            if (mlir::isa<VPU::ConcatOp>(producerOp) || (!mlir::isa<VPU::NCEClusterTilingOp>(producerOp) &&
                                                         !mlir::isa<VPU::ClusteredOpInterface>(producerOp))) {
                return false;
            }

            if (mlir::isa<VPU::NCEClusterTilingOp>(producerOp)) {
                auto clusterTilingOp = mlir::cast<VPU::NCEClusterTilingOp>(producerOp);
                auto distributedIf = clusterTilingOp.getResult(0).getType().dyn_cast<VPU::DistributedTypeInterface>();
                // For inner copyOut op
                if ((distributedIf == nullptr || !distributedIf.containsDistributedTypes()) &&
                    (clusterTilingOp.getInnerTaskOpOfType<VPU::CopyOp>() != nullptr)) {
                    distributedIf =
                            clusterTilingOp.getOperands()[0].getType().dyn_cast<VPU::DistributedTypeInterface>();
                }
                VPUX_THROW_WHEN(distributedIf == nullptr || !distributedIf.containsDistributedTypes(),
                                "distributedTensorType should not be nullptr for NCEClusterTilingOp");

                auto distributedTensorType =
                        distributedIf.getDistributedTypes().front().cast<VPU::DistributedTensorType>();
                auto mode = distributedTensorType.getDistribution().getMode().getValue();
                if (!VPU::bitEnumContainsAny(mode, DistributionMode::DUPLICATED) &&
                    !VPU::bitEnumContainsAny(mode, DistributionMode::MULTICASTED)) {
                    return false;
                }
            }

            if (mlir::isa<VPU::ClusteredOpInterface>(producerOp)) {
                auto clusteredProducer = mlir::cast<VPU::ClusteredOpInterface>(producerOp);
                const auto producerStrategy = clusteredProducer.getMultiClusterStrategy();
                if (!producerStrategy.has_value()) {
                    return false;
                }
                auto mode = VPU::getOutputTensorDistributionMode(clusteredProducer, producerStrategy.value(), nullptr);
                if (!VPU::bitEnumContainsAny(mode, DistributionMode::DUPLICATED) &&
                    !VPU::bitEnumContainsAny(mode, DistributionMode::MULTICASTED)) {
                    return false;
                }
            }
            auto eltwiseOp = mlir::dyn_cast<VPU::NCEEltwiseOp>(op);
            if (eltwiseOp == nullptr) {
                Logger::global().trace("Select DUPLICATED mode for the activation of SOH-like strategys");
                return true;
            }
            if (eltwiseOp.getIsInplace().value_or(false)) {
                // E135492: Accuracy issue with duplicated input for SOH-like inplace eltwise
                // TODO remove this check after the issue is fixed
                Logger::global().trace(
                        "Select SEGMENTED mode for the activation of SOH-like strategies for ELTWISE op");
                return false;
            }

            eltwiseInputsCompatible[operand.index()] = true;
        }

        if (std::all_of(eltwiseInputsCompatible.begin(), eltwiseInputsCompatible.end(), [](auto val) {
                return val;
            })) {
            Logger::global().trace("Select DUPLICATED mode for the activation of SOH-like strategys");
            return true;
        }

        return false;
    };

    if (strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped) {
        return isDuplicatedModeForSOHLikeStrategy() ? DistributionMode::DUPLICATED : DistributionMode::OVERLAPPED;
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverWidth) {
        return DistributionMode::OVERLAPPED;
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
               strategy == VPU::MultiClusterStrategy::HKSwitch) {
        // TODO: be more explicit ahead of time wrt MultiClusterStrategy for 40XX.
        // E#71926 to track this.
        if (VPU::isArchVPUX3XXX(VPU::getArch(clusteredOp))) {
            return DistributionMode::SEGMENTED;
        }
        return isDuplicatedModeForSOHLikeStrategy() ? DistributionMode::DUPLICATED : DistributionMode::OVERLAPPED;
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        if (isSegmentedInputCompatible(clusteredOp.getOperation())) {
            return DistributionMode::SEGMENTED;
        }
        return DistributionMode::DUPLICATED;
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverBatch) {
        return DistributionMode::SEGMENTED;
    } else if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return DistributionMode::DUPLICATED;
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverGroup) {
        return DistributionMode::SEGMENTED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "activation tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getWeightsTensorDistributionMode(VPU::MultiClusterStrategy strategy) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
        strategy == VPU::MultiClusterStrategy::SplitOverHeight || strategy == VPU::MultiClusterStrategy::Clustering ||
        strategy == VPU::MultiClusterStrategy::HKSwitch || strategy == VPU::MultiClusterStrategy::SplitOverBatch) {
        return DistributionMode::DUPLICATED;
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        return DistributionMode::SEGMENTED;
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverGroup) {
        return DistributionMode::SEGMENTED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "weights tensor",
                   strategy);
    }
}

DistributionMode vpux::VPU::getOutputTensorDistributionMode(VPU::ClusteredOpInterface clusteredOp,
                                                            VPU::MultiClusterStrategy strategy,
                                                            vpux::NDTypeInterface outputType) {
    if (strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped ||
        strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
        strategy == VPU::MultiClusterStrategy::SplitOverWidth) {
        // TODO: be more explicit ahead of time wrt MultiClusterStrategy for 40XX.
        // E#71926 to track this.
        if (VPU::isArchVPUX3XXX(VPU::getArch(clusteredOp)) || mlir::isa<SWOpInterface>(clusteredOp.getOperation())) {
            return DistributionMode::SEGMENTED;
        }
        return DistributionMode::OVERLAPPED;
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        const auto outputShape = outputType != nullptr
                                         ? outputType.getShape()
                                         : clusteredOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape();
        auto outputChannel = outputShape[Dims4D::Act::C];
        if (outputChannel > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
            return DistributionMode::SEGMENTED;
        }
        if (isSOKSegmentedOutputCompatible(clusteredOp.getOperation())) {
            return DistributionMode::SEGMENTED;
        }
        return DistributionMode::DUPLICATED | DistributionMode::SEGMENTED;
    } else if (strategy == VPU::MultiClusterStrategy::HKSwitch) {
        return DistributionMode::MULTICASTED | DistributionMode::SEGMENTED;
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverBatch) {
        return DistributionMode::SEGMENTED;
    } else if (strategy == VPU::MultiClusterStrategy::Clustering) {
        return DistributionMode::DUPLICATED;
    } else if (strategy == VPU::MultiClusterStrategy::SplitOverGroup) {
        return DistributionMode::SEGMENTED;
    } else {
        VPUX_THROW("{0} is an invalid multi-cluster strategy, unable to determine the distribution mode for the "
                   "output tensor",
                   strategy);
    }
}

// W * h_per_cluster has to be divisible by 4 or 8, if the input is sparse
// Based on the given width, the height alignment is computed and returned
// Note: For sparse inputs, the segment size has to be divisible by 8 to satisfy the segment size requirements for
// sparse inputs, more explicitly the requirements of the sp_seg_size register
int64_t vpux::VPU::getSOHPerClusterHeightAlignment(int64_t inputWidth, bool isInputSparse) {
    const auto spatialAlignment =
            isInputSparse ? VPU::NCEInvariant::VPU_SEGMENT_SIZE_SPARSE : VPU::NCEInvariant::VPU_SEGMENT_SIZE_DENSE;
    for (auto widthAlignment = spatialAlignment; widthAlignment >= 1; widthAlignment /= 2) {
        if (inputWidth % widthAlignment == 0) {
            return spatialAlignment / widthAlignment;
        }
    }
    return spatialAlignment;
}

int64_t vpux::VPU::getSOHMinimalHeightAlignment(vpux::ShapeRef shape, int64_t numClusters, bool isInputSparse,
                                                VPU::ArchKind arch) {
    if (!VPU::isArchVPUX3XXX(arch)) {
        return 1;
    }

    if (shape.size() < checked_cast<size_t>(Dims4D::Act::W.ind() + 1)) {
        return 1;
    }

    const auto spatialAlignment =
            isInputSparse ? VPU::NCEInvariant::VPU_SEGMENT_SIZE_SPARSE : VPU::NCEInvariant::VPU_SEGMENT_SIZE_DENSE;
    auto heightAlignment = getSOHPerClusterHeightAlignment(shape[Dims4D::Act::W], isInputSparse);
    for (int64_t alignment = 1; alignment < heightAlignment; alignment *= 2) {
        const auto hPerCluster = alignValUp(divUp(shape[Dims4D::Act::H], numClusters), alignment);
        if (hPerCluster * shape[Dims4D::Act::W] % spatialAlignment == 0) {
            heightAlignment = alignment;
            break;
        }
    }
    return heightAlignment;
}

// When doing SOH not all combinations are supported by HW in terms of how input is segmented
// Following rules need to be satisfied:
// - height of clusters from 0 to N - 1 must be equal
// - height of last cluster (which stores the remainder) must be <= of height of previous clusters
// - Width * height_per_cluster (for cluster 0 - N-1) must be multiple of 4 (or 8 for sparse inputs)
bool vpux::VPU::isSOHSupportedByDPU(vpux::NDTypeInterface inputType, ShapeRef inputShape, int64_t numClusters, bool,
                                    ArchKind arch) {
    // Layers with 5D input shapes does not support SOH
    if (inputShape.size() == DimsGroups5D::Act::numDims) {
        return false;
    }

    auto sparseInputType = inputType.dyn_cast<VPU::SparseTensorType>();
    const auto isInputSparse = sparseInputType != nullptr;
    if (isInputSparse) {
        auto inputDataShape = sparseInputType.getData().cast<vpux::NDTypeInterface>().getShape();
        // The input could be sparse with the data smaller than the storage element table
        // In that case, the SOH segments are created based on the table
        // If the data has fewer lines than the number of clusters, more clusters would read the data from other
        // clusters, resulting in numerous ISI reads which would affect the performance
        if (inputDataShape.size() == 4 && inputDataShape[Dims4D::Act::H] < numClusters) {
            return false;
        }
    }

    // On VPUX40XX, SOH doesn't have the rules above
    // Actually the input tile shapes are completely back-inferred by output tile shapes which are following
    // uniformDistributedSegments method
    const std::set<VPU::ArchKind> compatibleTargets = {
            VPU::ArchKind::NPU40XX,
    };
    if (compatibleTargets.count(arch) > 0) {
        return true;
    }

    auto IH = inputShape[Dims4D::Act::H];
    auto IW = inputShape[Dims4D::Act::W];

    auto hPerCluster = divUp(IH, numClusters);
    auto alignment = getSOHPerClusterHeightAlignment(IW, isInputSparse);

    hPerCluster = alignValUp(hPerCluster, alignment);

    auto hLastCluster = IH - hPerCluster * (numClusters - 1);

    return (hLastCluster > 0);
}

bool vpux::VPU::isSOGSupportedByDPU([[maybe_unused]] vpux::NDTypeInterface inputType,
                                    [[maybe_unused]] ShapeRef inputShape, [[maybe_unused]] int64_t numClusters,
                                    [[maybe_unused]] bool DWTypeOp, [[maybe_unused]] ArchKind arch) {
    return true;
}

int64_t vpux::VPU::getOptimalNumClusters(mlir::Operation* operation, ShapeRef outputShape,
                                         VPU::MultiClusterStrategy strategy) {
    auto module = operation->getParentOfType<mlir::ModuleOp>();

    // Both ACT Shaves and DPUs are grouped together in NCE clusters, in a symmetric manner.
    // For VPUX37XX and subsequent, each NCE cluster 1 DPU and 2 ACT shaves.
    // Thus shaves have the availability for distributing across clusters similar to DPUs.
    auto numClustersAvailableForCompilation = IE::getTileExecutor(module).getCount();
    auto optimalNumberOfClusters = numClustersAvailableForCompilation;

    // Here the number of clusters to be used for an individual SOK layer is determined
    // such that additional alignment of the per cluster output channels is not required.
    // For example 80 output channels, the weights should only be split on 3 clusters [32, 32, 16].
    // Also when creating the copy-in for the activation we need to ensure that the number
    // of clusters that the input is duplicated to is also 3 clusters in this case.
    // Therefore we use the variable optimalNumberOfClusters for both purposes here, to determine
    // num_tiles and numClusters for the activations and the weights.
    if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
        const auto OC = outputShape[Dims4D::Act::C];
        int64_t numClustersToUseForLayer = numClustersAvailableForCompilation;
        if (mlir::isa<VPU::DequantizeOp>(operation)) {
            // E#149000 : Proper implementation should be done for Dequantize with correct alignment
            const auto dequantizeOC = outputShape[Dims4D::Act::N];
            numClustersToUseForLayer = std::min(numClustersToUseForLayer, dequantizeOC);
        } else if (mlir::isa<VPU::SWOpInterface>(operation)) {
            numClustersToUseForLayer = std::min(numClustersToUseForLayer, OC);
        } else {
            auto uniformDistributedSegments = VPU::isUniformDistributedSegmentsSupported(operation);
            numClustersToUseForLayer =
                    getNumberOfClustersForSOKToAvoidAlignment(OC, numClustersToUseForLayer, uniformDistributedSegments);

            if (mlir::isa<VPU::ConcatOp>(operation) && numClustersToUseForLayer == 1) {
                numClustersToUseForLayer = std::min(numClustersAvailableForCompilation, OC);
            }
        }
        optimalNumberOfClusters = numClustersToUseForLayer;
    }

    // Limit number clusters to batch size for SOB.
    if (strategy == VPU::MultiClusterStrategy::SplitOverBatch) {
        const int64_t maxNumClusters = outputShape[Dims4D::Act::N];
        optimalNumberOfClusters = std::min(maxNumClusters, numClustersAvailableForCompilation);
    }
    return optimalNumberOfClusters;
}

bool vpux::VPU::getUniformDistributedSegments(VPU::ClusteredOpInterface clusteredOp, ArrayRef<int64_t> shape,
                                              VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
                                              ArrayRef<int64_t> alignment) {
    if (!VPU::isUniformDistributedSegmentsSupported(clusteredOp.getOperation())) {
        return false;
    }

    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(clusteredOp.getOperation());
    if (nceOp == nullptr) {
        return true;
    }

    if (!nceOp->getResult(0).getType().isa<VPU::SparseTensorType>()) {
        return true;
    }

    if (!VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::SEGMENTED)) {
        return true;
    }

    VPUX_THROW_WHEN(numTiles.empty(), "numTiles cannot be nullptr for distribution mode = {0}", distributionMode);

    const auto axis = vpux::VPU::getDistributedTilingAxis(numTiles);

    // If NCE op with Sparse output is not SOK, let segmentation be done uniformly
    if (axis != Dims4D::Act::C.ind() && axis != Dims4D::Filter::OC.ind()) {
        return true;
    }

    // For a SOK layer with sparse output, try not using uniformDistributedSegments because NCE operations with sparse
    // outputs must have all variants with the same number of channels excluding the last one
    SmallVector<int64_t> tiledShape(shape);
    SmallVector<int64_t> remainderTileShape(shape);
    // Split in an equal manner such that first N-1 tiles are equal
    // and the last tile can be less or equal.
    tiledShape[axis] = divUp(tiledShape[axis], numTiles[axis]);

    if (!alignment.empty()) {
        tiledShape = alignShape(tiledShape, alignment, alignValUp<int64_t>);
    }

    remainderTileShape[axis] = shape[axis] - tiledShape[axis] * (numTiles[axis] - 1);
    if (remainderTileShape[axis] > 0) {
        return false;
    }

    return true;
}

VPU::DistributedTensorType vpux::VPU::createExplicitDistributedTensorType(
        VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface inputType, DistributionMode distributionMode,
        ArrayRef<int64_t> numTiles, int64_t numClusters, ArrayRef<int64_t> alignment,
        const bool uniformDistributedSegments, const VPU::OverlapDistributionParams& overlapParams) {
    auto ctx = clusteredOp->getContext();

    const auto memSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(MemoryKind::CMX_NN));

    const auto order = mlir::AffineMapAttr::get(inputType.getDimsOrder().toAffineMap(ctx));
    auto elemType = inputType.getElementType();

    return DistributedTensorType::get(
            ctx, inputType.getShape().raw(), elemType, order, memSpace,
            VPU::DistributionInfo::getAttrFromClass(
                    ctx, clusteredOp.getExplicitDistributionInfoAttr(inputType.getShape(), distributionMode, numTiles,
                                                                     numClusters, alignment, uniformDistributedSegments,
                                                                     overlapParams)));
}

VPU::DistributedTensorType vpux::VPU::createDistributedTensorType(
        VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface inputType, DistributionMode distributionMode,
        ArrayRef<int64_t> numTiles, int64_t numClusters, ArrayRef<int64_t> alignment,
        const bool uniformDistributedSegments, const bool hasExplicitDistributionInfoAttribute,
        const VPU::OverlapDistributionParams& overlapParams) {
    if (hasExplicitDistributionInfoAttribute || overlapParams.hasNonnullComputeAndMemoryShapesOffsets()) {
        numTiles = (VPU::bitEnumContainsAny(distributionMode, DistributionMode::OVERLAPPED) ||
                    VPU::bitEnumContainsAny(distributionMode, DistributionMode::SEGMENTED))
                           ? numTiles
                           : ArrayRef<int64_t>{};
        return createExplicitDistributedTensorType(clusteredOp, inputType, distributionMode, numTiles, numClusters,
                                                   alignment, uniformDistributedSegments, overlapParams);
    }

    return llvm::TypeSwitch<mlir::Operation*, DistributedTensorType>(clusteredOp.getOperation())
            .Case<VPU::SWOpInterface>([&](VPU::SWOpInterface swOp) {
                return createDistributedTensorType(swOp, inputType, distributionMode, numTiles, numClusters, alignment,
                                                   uniformDistributedSegments);
            })
            .Case<VPU::NCEOpInterface>([&](VPU::NCEOpInterface nceOp) {
                auto padAttr =
                        overlapParams.getPads().has_value()
                                ? VPU::Padding::getAttrFromClass(nceOp.getContext(), overlapParams.getPads().value())
                                : nullptr;

                return createDistributedTensorType(nceOp, inputType, distributionMode, numTiles, numClusters, alignment,
                                                   uniformDistributedSegments, overlapParams.getKernel(), padAttr,
                                                   overlapParams.getStride(),
                                                   overlapParams.hasEqualComputeAndMemoryView());
            })
            .Case<VPU::ConcatOp>([&](VPU::ConcatOp concatOp) {
                auto padAttr =
                        overlapParams.getPads().has_value()
                                ? VPU::Padding::getAttrFromClass(concatOp.getContext(), overlapParams.getPads().value())
                                : nullptr;

                return createDistributedTensorType(concatOp.getOperation(), inputType, distributionMode, numTiles,
                                                   numClusters, alignment, uniformDistributedSegments,
                                                   overlapParams.getKernel(), padAttr, overlapParams.getStride());
            })
            .Default([clusteredOp](mlir::Operation*) -> DistributedTensorType {
                VPUX_THROW("unsupported operation for createDistributedTensorType: {0}", clusteredOp);
            });
}

VPU::SparseTensorType vpux::VPU::createSparseTensorDistributedType(
        VPU::ClusteredOpInterface clusteredOp, VPU::SparseTensorType sparseInputType, DistributionMode distributionMode,
        ArrayRef<int64_t> numTiles, int64_t numClusters, ArrayRef<int64_t> alignment,
        const bool uniformDistributedSegments, const bool hasExplicitDistributedAttr,
        const VPU::OverlapDistributionParams& overlapParams) {
    auto* ctx = clusteredOp.getContext();

    VPUX_THROW_WHEN(sparseInputType.getSparsityMap() == nullptr, "Missing input sparsity map");

    const auto dataType = sparseInputType.getData().cast<vpux::NDTypeInterface>();
    const auto storageElementTable = sparseInputType.getStorageElementTable();
    if (storageElementTable == nullptr) {
        const auto distributedDataType =
                createDistributedTensorType(clusteredOp, dataType, distributionMode, numTiles, numClusters, alignment,
                                            uniformDistributedSegments, hasExplicitDistributedAttr, overlapParams);
        const auto distributedSMType = createDistributedTensorType(
                clusteredOp, sparseInputType.getSparsityMap().cast<vpux::NDTypeInterface>(), distributionMode, numTiles,
                numClusters, alignment, uniformDistributedSegments, hasExplicitDistributedAttr, overlapParams);

        return VPU::SparseTensorType::get(distributedDataType, distributedSMType, nullptr,
                                          sparseInputType.getIsWeights(), sparseInputType.getSparsityCompression(),
                                          sparseInputType.getSeAttr());
    }

    auto seTableAlignmentArr = SmallVector<int64_t>(alignment);
    if (!alignment.empty()) {
        seTableAlignmentArr[Dims4D::Act::C.ind()] = 1;
    }

    // The input data has no alignment requirement when the SE table is present
    auto dataAlignmentArr = SmallVector<int64_t>{};
    if (!hasExplicitDistributedAttr && !overlapParams.hasNonnullComputeAndMemoryShapesOffsets()) {
        VPUX_THROW_WHEN(distributionMode == VPU::DistributionMode::OVERLAPPED,
                        "Sparse type has StorageElementTable and OVERLAPPED mode should enable explicit "
                        "distributed attribution");
        const auto distributedDataType = createDistributedTensorType(
                clusteredOp, dataType, distributionMode, numTiles, numClusters, dataAlignmentArr,
                uniformDistributedSegments, hasExplicitDistributedAttr, overlapParams);
        const auto distributedSMType = createDistributedTensorType(
                clusteredOp, sparseInputType.getSparsityMap().cast<vpux::NDTypeInterface>(), distributionMode, numTiles,
                numClusters, alignment, uniformDistributedSegments, hasExplicitDistributedAttr, overlapParams);
        const auto distributedSEType = createDistributedTensorType(
                clusteredOp, storageElementTable.cast<vpux::NDTypeInterface>(), distributionMode, numTiles, numClusters,
                seTableAlignmentArr, uniformDistributedSegments, hasExplicitDistributedAttr, overlapParams);

        return VPU::SparseTensorType::get(distributedDataType, distributedSMType, distributedSEType,
                                          sparseInputType.getIsWeights(), sparseInputType.getSparsityCompression(),
                                          sparseInputType.getSeAttr());
    }

    auto effectiveSparseType = VPU::getEffectiveSparseOutputType(sparseInputType);
    auto distributedEffectiveData = createDistributedTensorType(
            clusteredOp, effectiveSparseType, distributionMode, numTiles, numClusters, alignment,
            uniformDistributedSegments, hasExplicitDistributedAttr, overlapParams);
    const auto effectiveDataDistribution = distributedEffectiveData.getDistribution();

    auto dataDistribution = getExplicitDistrAttrForSparseData(effectiveDataDistribution, dataType.getShape(),
                                                              sparseInputType.getSeAttr(), ctx);
    const auto distributedDataType = VPU::DistributedTensorType::get(
            ctx, dataType.getShape().raw(), distributedEffectiveData.getElementType(),
            distributedEffectiveData.getOrder(), distributedEffectiveData.getMemSpace(), dataDistribution);

    const auto smType = sparseInputType.getSparsityMap().cast<vpux::NDTypeInterface>();
    const auto smDistribution = getExplicitDistrAttrForSparsityMap(effectiveDataDistribution, smType.getShape(),
                                                                   sparseInputType.getIsWeights(), ctx);
    const auto distributedSMType = VPU::DistributedTensorType::get(
            ctx, smType.getShape().raw(), smType.getElementType(), distributedEffectiveData.getOrder(),
            distributedEffectiveData.getMemSpace(), smDistribution);

    const auto seType = storageElementTable.cast<vpux::NDTypeInterface>();
    auto seDistribution = getExplicitDistrAttrForSETable(
            effectiveDataDistribution, smType.getShape()[Dims4D::Act::C] / seType.getShape()[Dims4D::Act::C], ctx);
    const auto distributedSEType = VPU::DistributedTensorType::get(
            ctx, seType.getShape().raw(), seType.getElementType(), distributedEffectiveData.getOrder(),
            distributedEffectiveData.getMemSpace(), seDistribution);

    return VPU::SparseTensorType::get(distributedDataType, distributedSMType, distributedSEType,
                                      sparseInputType.getIsWeights(), sparseInputType.getSparsityCompression(),
                                      sparseInputType.getSeAttr());
}

DistributedTensorType vpux::VPU::createDistributedTensorType(VPU::SWOpInterface swOp, vpux::NDTypeInterface inputType,
                                                             DistributionMode distributionMode,
                                                             ArrayRef<int64_t> numTiles,
                                                             int64_t optimalNumberOfClusters,
                                                             ArrayRef<int64_t> alignment,
                                                             const bool uniformDistributedSegments) {
    auto* ctx = swOp->getContext();
    const auto memSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(MemoryKind::CMX_NN));

    const auto order = mlir::AffineMapAttr::get(inputType.getDimsOrder().toAffineMap(ctx));
    auto elemType = inputType.getElementType();

    return DistributedTensorType::get(
            ctx, inputType.getShape().raw(), elemType, order, memSpace,
            VPU::DistributionInfo::getAttrFromClass(
                    ctx, createDistributionInfo(swOp, distributionMode, numTiles, optimalNumberOfClusters, alignment,
                                                uniformDistributedSegments)));
}

DistributedTensorType vpux::VPU::createDistributedTensorType(
        VPU::NCEOpInterface nceOp, vpux::NDTypeInterface inputType, DistributionMode distributionMode,
        ArrayRef<int64_t> numTiles, int64_t optimalNumberOfClusters, ArrayRef<int64_t> alignment,
        const bool uniformDistributedSegments, ArrayRef<int64_t> kernel, VPU::PaddingAttr pad, ArrayRef<int64_t> stride,
        const bool equalComputeAndMemoryView) {
    auto* ctx = nceOp->getContext();

    const auto shape = inputType.getShape();
    const auto memSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(MemoryKind::CMX_NN));

    const auto order = mlir::AffineMapAttr::get(inputType.getDimsOrder().toAffineMap(ctx));
    auto elemType = inputType.getElementType();

    return DistributedTensorType::get(
            ctx, shape.raw(), elemType, order, memSpace,
            VPU::DistributionInfo::getAttrFromClass(
                    ctx, createDistributionInfo(nceOp, distributionMode, numTiles, optimalNumberOfClusters, alignment,
                                                uniformDistributedSegments, kernel, VPU::Padding::getClassFromAttr(pad),
                                                stride, equalComputeAndMemoryView)));
}

DistributedTensorType vpux::VPU::createDistributedTensorType(
        mlir::Operation* viewLikeOp, vpux::NDTypeInterface inputType, DistributionMode distributionMode,
        ArrayRef<int64_t> numTiles, int64_t optimalNumberOfClusters, ArrayRef<int64_t> alignment,
        const bool uniformDistributedSegments, ArrayRef<int64_t> kernel, VPU::PaddingAttr pad,
        ArrayRef<int64_t> stride) {
    VPUX_THROW_UNLESS(mlir::isa_and_nonnull<VPU::ViewLikeOpInterface>(viewLikeOp), "Op {0} is not a view like op",
                      viewLikeOp->getName());
    auto* ctx = viewLikeOp->getContext();

    const auto memSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(MemoryKind::CMX_NN));

    const auto order = mlir::AffineMapAttr::get(inputType.getDimsOrder().toAffineMap(ctx));
    auto elemType = inputType.getElementType();

    return DistributedTensorType::get(
            ctx, inputType.getShape().raw(), elemType, order, memSpace,
            VPU::DistributionInfo::getAttrFromClass(
                    ctx, createDistributionInfo(viewLikeOp, distributionMode, numTiles, optimalNumberOfClusters,
                                                alignment, uniformDistributedSegments, kernel,
                                                VPU::Padding::getClassFromAttr(pad), stride)));
}

vpux::VPU::CopyOp vpux::VPU::createDistributedCopyIn(mlir::PatternRewriter& rewriter,
                                                     VPU::ClusteredOpInterface clusteredOp, mlir::Value input,
                                                     vpux::NDTypeInterface inputTensorDistributedTensorType) {
    rewriter.setInsertionPoint(clusteredOp);
    const auto memSpace = IndexedSymbolAttr::get(rewriter.getContext(), stringifyEnum(MemoryKind::CMX_NN));
    auto distributedInputCopyOp =
            rewriter.create<VPU::CopyOp>(clusteredOp.getLoc(), inputTensorDistributedTensorType, input, memSpace);

    return distributedInputCopyOp;
}

vpux::VPU::UnrolledTypeOp vpux::VPU::createDistributedUnrolledTypeIn(
        mlir::PatternRewriter& rewriter, VPU::ClusteredOpInterface clusteredOp, mlir::Value input,
        vpux::NDTypeInterface inputTensorDistributedTensorType) {
    rewriter.setInsertionPoint(clusteredOp);
    auto distributedInputCopyOp =
            rewriter.create<VPU::UnrolledTypeOp>(clusteredOp.getLoc(), inputTensorDistributedTensorType, input);

    return distributedInputCopyOp;
}

vpux::NDTypeInterface vpux::VPU::getDistributedTypeFromInput(VPU::ClusteredOpInterface clusteredOp, mlir::Value input,
                                                             DistributionMode distributionMode,
                                                             mlir::ArrayAttr numTiles, mlir::ArrayAttr alignment,
                                                             VPU::MultiClusterStrategy strategy,
                                                             const bool hasExplicitDistributedAttr,
                                                             SiblingOpsAnalysis& siblingsAnalysis) {
    const auto outputType = clusteredOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto numClusters = getOptimalNumClusters(clusteredOp, outputType.getShape(), strategy);

    auto ndTypeInterfaceInput = input.getType().cast<vpux::NDTypeInterface>();

    const auto numTilesArr = numTiles ? parseIntArrayAttr<int64_t>(numTiles) : SmallVector<int64_t>{};
    const auto alignmentArr = alignment ? parseIntArrayAttr<int64_t>(alignment) : SmallVector<int64_t>{};
    auto uniformDistributedSegments = VPU::getUniformDistributedSegments(
            clusteredOp, ndTypeInterfaceInput.getShape().raw(), distributionMode, numTilesArr, alignmentArr);

    auto getOverlappedParams = [&]() -> OverlapDistributionParams {
        if (distributionMode != DistributionMode::OVERLAPPED) {
            return OverlapDistributionParams();
        }

        auto swOp = mlir::dyn_cast<VPU::SWOpInterface>(clusteredOp.getOperation());
        if (swOp == nullptr) {
            return getActivationOverlappedParams(clusteredOp, numTilesArr, uniformDistributedSegments,
                                                 siblingsAnalysis);
        }

        auto outputShape = swOp->getResult(0).getType().cast<NDTypeInterface>().getShape();
        return getExplicitOverlapParamsForSWOpInput(swOp, outputShape, numTilesArr, alignmentArr);
    };

    const OverlapDistributionParams overlappedParams = getOverlappedParams();

    vpux::NDTypeInterface inputTensorDistributedTensorType;
    if (auto sparseInputType = input.getType().dyn_cast<VPU::SparseTensorType>()) {
        inputTensorDistributedTensorType = createSparseTensorDistributedType(
                clusteredOp, sparseInputType, distributionMode, numTilesArr, numClusters, alignmentArr,
                uniformDistributedSegments, hasExplicitDistributedAttr, overlappedParams);
    } else {
        inputTensorDistributedTensorType = createDistributedTensorType(
                clusteredOp, ndTypeInterfaceInput, distributionMode, numTilesArr, numClusters, alignmentArr,
                uniformDistributedSegments, hasExplicitDistributedAttr, overlappedParams);
    }

    return inputTensorDistributedTensorType;
}

VPU::DistributedTypeInterface vpux::VPU::getDistributedActivationTypeFromOp(VPU::ClusteredOpInterface clusteredOp,
                                                                            vpux::NDTypeInterface inputType,
                                                                            int64_t numClusters,
                                                                            vpux::NDTypeInterface tiledOutputType,
                                                                            const vpux::TileInfo& tileInfo) {
    VPUX_THROW_UNLESS(clusteredOp.getMultiClusterStrategy().has_value(),
                      "Op {0} does not have multiClusterStrategy attribute", clusteredOp->getLoc());
    return getDistributedActivationTypeFromOp(clusteredOp, inputType, numClusters,
                                              clusteredOp.getMultiClusterStrategy().value(),
                                              /*customAlignment*/ ArrayRef<int64_t>{}, tiledOutputType, tileInfo);
}

VPU::DistributedTypeInterface vpux::VPU::getDistributedActivationTypeFromOp(
        VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface inputType, int64_t numClusters,
        VPU::MultiClusterStrategy customStrategy, ArrayRef<int64_t> customAlignment,
        vpux::NDTypeInterface tiledOutputType, const vpux::TileInfo& tileInfo) {
    auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, customStrategy);
    auto activationTensorNumTiles = getActivationTensorNumTiles(clusteredOp, numClusters, customStrategy, inputType);
    if (mlir::isa<VPU::SWOpInterface>(clusteredOp.getOperation())) {
        activationTensorDistributionMode = getSWInputTensorDistributionMode(clusteredOp, customStrategy, inputType);
        activationTensorNumTiles = getSWInputTensorNumTiles(clusteredOp, numClusters, customStrategy, inputType);
    }

    auto actualOutputType =
            tiledOutputType != nullptr ? tiledOutputType : clusteredOp->getResult(0).getType().cast<NDTypeInterface>();

    auto customAlignmentArr = SmallVector<int64_t>{};
    if (customAlignment.empty()) {
        const auto activationAlignment =
                getActivationTensorAlignment(clusteredOp, numClusters, customStrategy, inputType, actualOutputType);
        if (activationAlignment.has_value()) {
            customAlignmentArr = activationAlignment.value();
        }
    }

    auto uniformDistributedSegments = VPU::getUniformDistributedSegments(clusteredOp, inputType.getShape().raw(),
                                                                         activationTensorDistributionMode,
                                                                         activationTensorNumTiles, customAlignmentArr);

    auto getOverlappedParams = [&]() -> OverlapDistributionParams {
        if (activationTensorDistributionMode != DistributionMode::OVERLAPPED) {
            return OverlapDistributionParams();
        }

        auto swOp = mlir::dyn_cast<VPU::SWOpInterface>(clusteredOp.getOperation());
        if (swOp == nullptr) {
            return getActivationOverlappedParams(clusteredOp, activationTensorNumTiles, uniformDistributedSegments,
                                                 inputType, tileInfo);
        }

        return getExplicitOverlapParamsForSWOpInput(swOp, actualOutputType.getShape(), activationTensorNumTiles,
                                                    customAlignmentArr);
    };

    const OverlapDistributionParams overlappedParams = getOverlappedParams();

    const auto hasExplicitDistributedAttr = overlappedParams.hasNonnullComputeAndMemoryShapesOffsets();

    if (auto sparseType = inputType.dyn_cast<VPU::SparseTensorType>()) {
        return createSparseTensorDistributedType(
                clusteredOp, sparseType, activationTensorDistributionMode, activationTensorNumTiles, numClusters,
                customAlignmentArr, uniformDistributedSegments, hasExplicitDistributedAttr, overlappedParams);
    }

    return createDistributedTensorType(clusteredOp, inputType, activationTensorDistributionMode,
                                       activationTensorNumTiles, numClusters, customAlignmentArr,
                                       uniformDistributedSegments, hasExplicitDistributedAttr, overlappedParams);
}

VPU::DistributedTypeInterface vpux::VPU::getDistributedFilterTypeFromOp(VPU::NCEOpInterface nceOp,
                                                                        vpux::NDTypeInterface inputType,
                                                                        int64_t numClusters) {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp.getMultiClusterStrategy().has_value(),
                      "Op {0} does not have multiClusterStrategy attribute", nceOp->getLoc());
    return getDistributedFilterTypeFromOp(nceOp, inputType, numClusters, clusteredOp.getMultiClusterStrategy().value());
}

VPU::DistributedTypeInterface vpux::VPU::getDistributedFilterTypeFromOp(VPU::NCEOpInterface nceOp,
                                                                        vpux::NDTypeInterface inputType,
                                                                        int64_t numClusters,
                                                                        VPU::MultiClusterStrategy customStrategy) {
    auto weightAlignmentArr = SmallVector<int64_t>{};
    const auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(customStrategy);
    const auto weightsTensorNumTiles = getWeightsTensorNumTiles(clusteredOp, inputType, numClusters, customStrategy);

    const auto weightAlignment = getWeightsTensorAlignment(customStrategy);

    if (weightAlignment.has_value()) {
        weightAlignmentArr = weightAlignment.value();
    }

    auto uniformDistributedSegments = VPU::getUniformDistributedSegments(
            mlir::cast<VPU::ClusteredOpInterface>(nceOp.getOperation()), inputType.getShape().raw(),
            weightsTensorDistributionMode, weightsTensorNumTiles, weightAlignmentArr);

    if (auto sparseType = inputType.dyn_cast<VPU::SparseTensorType>()) {
        VPUX_THROW_UNLESS(sparseType.getSparsityMap() != nullptr, "Missing filter sparsity map");
        auto distributedDataType = createDistributedTensorType(
                nceOp, sparseType.getData().cast<vpux::NDTypeInterface>(), weightsTensorDistributionMode,
                weightsTensorNumTiles, numClusters, weightAlignmentArr, uniformDistributedSegments);
        auto distributedSMType = createDistributedTensorType(
                nceOp, sparseType.getSparsityMap().cast<vpux::NDTypeInterface>(), weightsTensorDistributionMode,
                weightsTensorNumTiles, numClusters, weightAlignmentArr, uniformDistributedSegments);
        auto isWeights = mlir::UnitAttr::get(nceOp.getContext());
        return VPU::SparseTensorType::get(distributedDataType, distributedSMType, nullptr, isWeights,
                                          sparseType.getSparsityCompression());
    }

    return createDistributedTensorType(nceOp, inputType, weightsTensorDistributionMode, weightsTensorNumTiles,
                                       numClusters, weightAlignmentArr, uniformDistributedSegments);
}

VPU::DistributedTypeInterface vpux::VPU::getDistributedOutputTypeFromOp(
        VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface outputType, int64_t numClusters,
        ArrayRef<vpux::NDTypeInterface> inputTypes, const vpux::TileInfo& tileInfo,
        const bool hasExplicitDistributedAttr, const std::optional<OverlapDistributionParams>& overlappedParams) {
    VPUX_THROW_UNLESS(clusteredOp.getMultiClusterStrategy().has_value(),
                      "Op {0} does not have multiClusterStrategy attribute", clusteredOp->getLoc());
    return getDistributedOutputTypeFromOp(clusteredOp, outputType, numClusters,
                                          clusteredOp.getMultiClusterStrategy().value(), inputTypes, tileInfo,
                                          hasExplicitDistributedAttr, overlappedParams);
}

/**
 * Match the pattern SOHO_NCEPermute (SEGMENTED) -> SOHO_Conv
 * where the tensor should be converted to OVERLAPPED to avoid spilling
 */
bool isOverlapOutputPatternRequired(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy) {
    if (!mlir::isa<VPU::NCEPermuteOp>(clusteredOp.getOperation()) ||
        strategy != VPU::MultiClusterStrategy::SplitOverHeightOverlapped) {
        return false;
    }
    auto defaultOutputMode = getOutputTensorDistributionMode(clusteredOp, strategy, nullptr);
    if (defaultOutputMode != DistributionMode::SEGMENTED) {
        return false;
    }
    auto childOp = getNextCompressConv(clusteredOp.getOperation());
    return childOp != nullptr;
}

VPU::DistributedTypeInterface vpux::VPU::getDistributedOutputTypeFromOp(
        VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface outputType, int64_t numClusters,
        VPU::MultiClusterStrategy customStrategy, ArrayRef<vpux::NDTypeInterface> inputTypes,
        const vpux::TileInfo& tileInfo, const bool hasExplicitDistributedAttr,
        const std::optional<OverlapDistributionParams>& overlappedParamsOpt) {
    const auto outputTensorNumTiles = getOutputTensorNumTiles(clusteredOp, numClusters, customStrategy, outputType);
    // NCEPermute(SOHO) -> Conv(SOHO)
    // The output tensor of the NCEPermute should be OVERLAPPED to avoid spilling
    if (isOverlapOutputPatternRequired(clusteredOp, customStrategy)) {
        const auto origInputType = clusteredOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
        const auto nextConv = getNextCompressConv(clusteredOp.getOperation());
        auto inputDistType = getDistributedActivationTypeFromOp(clusteredOp, origInputType, numClusters, customStrategy)
                                     .cast<VPU::DistributedTensorType>();
        const auto fusedDistType = fuseOverlapParams(clusteredOp, inputDistType, nextConv, hasExplicitDistributedAttr);
        const OverlapDistributionParams permuteOverlapParams = {};
        const auto equalComputeAndMemoryView = true;
        auto distOutType =
                composeDistributedType(clusteredOp, fusedDistType.cast<VPU::DistributedTensorType>(), outputType,
                                       inputDistType.getDistribution().getNumTiles(), permuteOverlapParams,
                                       hasExplicitDistributedAttr, equalComputeAndMemoryView);

        return distOutType;
    }
    const auto outputTensorDistributionMode = getOutputTensorDistributionMode(clusteredOp, customStrategy, outputType);
    auto outputAlignmentArr = getOutAlignment(clusteredOp, numClusters, customStrategy, inputTypes, outputType);
    auto uniformDistributedSegments =
            VPU::getUniformDistributedSegments(clusteredOp, outputType.getShape().raw(), outputTensorDistributionMode,
                                               outputTensorNumTiles, outputAlignmentArr);

    OverlapDistributionParams overlappedParams;
    if (overlappedParamsOpt.has_value()) {
        overlappedParams = overlappedParamsOpt.value();
    } else {
        overlappedParams = (outputTensorDistributionMode == DistributionMode::OVERLAPPED &&
                            !mlir::isa<VPU::SWOpInterface>(clusteredOp.getOperation()))
                                   ? getOutputOverlappedParams(clusteredOp, outputTensorNumTiles,
                                                               uniformDistributedSegments, outputType, tileInfo)
                                   : OverlapDistributionParams();
    }

    if (auto sparseType = outputType.dyn_cast<VPU::SparseTensorType>()) {
        VPUX_THROW_UNLESS(sparseType.getSparsityMap() != nullptr, "Missing output sparsity map");
        VPUX_THROW_UNLESS(sparseType.getStorageElementTable() == nullptr,
                          "Storage element table is not supported for weights input");
        auto distributedDataType = createDistributedTensorType(
                clusteredOp, sparseType.getData().cast<vpux::NDTypeInterface>(), outputTensorDistributionMode,
                outputTensorNumTiles, numClusters, outputAlignmentArr, uniformDistributedSegments,
                hasExplicitDistributedAttr, overlappedParams);
        auto distributedSMType = createDistributedTensorType(
                clusteredOp, sparseType.getSparsityMap().cast<vpux::NDTypeInterface>(), outputTensorDistributionMode,
                outputTensorNumTiles, numClusters, outputAlignmentArr, uniformDistributedSegments,
                hasExplicitDistributedAttr, overlappedParams);
        return VPU::SparseTensorType::get(distributedDataType, distributedSMType);
    }

    return createDistributedTensorType(clusteredOp, outputType, outputTensorDistributionMode, outputTensorNumTiles,
                                       numClusters, outputAlignmentArr, uniformDistributedSegments,
                                       hasExplicitDistributedAttr, overlappedParams);
}

vpux::NDTypeInterface vpux::VPU::getDistributedOutputTensorType(
        VPU::ClusteredOpInterface clusteredOp, int64_t numClusters, VPU::MultiClusterStrategy strategy,
        vpux::NDTypeInterface outputTensorType, const bool hasExplicitDistributedAttr, bool alignForSOH,
        const std::optional<OverlapDistributionParams>& overlappedParams) {
    vpux::NDTypeInterface distributedOutputTensorType;
    if (auto sparseOutputType = outputTensorType.dyn_cast<VPU::SparseTensorType>()) {
        VPUX_THROW_UNLESS(sparseOutputType.getSparsityMap() != nullptr, "Missing sparsity map from sparse type {0}",
                          sparseOutputType);
        VPUX_THROW_UNLESS(sparseOutputType.getStorageElementTable() == nullptr,
                          "Dynamically populated storage element table is not supported");
        auto distributedDataType = getDistributedOutputTypeFromOp(clusteredOp, sparseOutputType.getData(), numClusters,
                                                                  /*inputTypes*/ {},
                                                                  /*tileInfo*/ TileInfo(ShapeRef()),
                                                                  hasExplicitDistributedAttr, overlappedParams);
        auto distributedSMType = getDistributedOutputTypeFromOp(
                clusteredOp, sparseOutputType.getSparsityMap(), numClusters,
                /*inputTypes*/ {}, /*tileInfo*/ TileInfo(ShapeRef()), hasExplicitDistributedAttr, overlappedParams);
        distributedOutputTensorType = VPU::SparseTensorType::get(distributedDataType, distributedSMType);
    } else {
        distributedOutputTensorType = getDistributedOutputTypeFromOp(clusteredOp, outputTensorType, numClusters,
                                                                     /*inputTypes*/ {},
                                                                     /*tileInfo*/ TileInfo(ShapeRef()),
                                                                     hasExplicitDistributedAttr, overlappedParams);
    }

    if (alignForSOH && strategy == VPU::MultiClusterStrategy::SplitOverHeight) {
        const auto newDistributedOutputTensorType =
                adjustOutputAlignmentForSOH(clusteredOp, distributedOutputTensorType);

        if (newDistributedOutputTensorType.has_value()) {
            distributedOutputTensorType = newDistributedOutputTensorType.value();
        }
    }

    return distributedOutputTensorType;
}

vpux::NDTypeInterface vpux::VPU::getDistributedOutputTensorType(VPU::ClusteredOpInterface clusteredOp,
                                                                vpux::NDTypeInterface outputTensorType,
                                                                SiblingOpsAnalysis& siblingsAnalysis,
                                                                VPU::MultiClusterStrategy strategy,
                                                                const bool hasExplicitDistributedAttr) {
    const auto outputTensorDistributionMode = getOutputTensorDistributionMode(clusteredOp, strategy, outputTensorType);
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape(), strategy);
    const auto outputTensorNumTiles = getOutputTensorNumTiles(clusteredOp, numClusters, strategy);
    auto outputAlignmentArr = getOutAlignment(clusteredOp, numClusters, strategy, {}, outputTensorType);
    auto uniformDistributedSegments =
            VPU::getUniformDistributedSegments(clusteredOp, outputTensorType.getShape().raw(),
                                               outputTensorDistributionMode, outputTensorNumTiles, outputAlignmentArr);

    const OverlapDistributionParams overlappedParams =
            (outputTensorDistributionMode == DistributionMode::OVERLAPPED &&
             !mlir::isa<VPU::SWOpInterface>(clusteredOp.getOperation()))
                    ? getOutputOverlappedParams(clusteredOp, outputTensorNumTiles, uniformDistributedSegments,
                                                outputTensorType, TileInfo(ShapeRef()), siblingsAnalysis)
                    : OverlapDistributionParams();

    return getDistributedOutputTensorType(clusteredOp, numClusters, strategy, outputTensorType,
                                          hasExplicitDistributedAttr, true, overlappedParams);
}

mlir::Type vpux::VPU::getCompactTypeFromDistributed(mlir::Type originalType) {
    auto compactType = originalType;

    if (auto distributedType = originalType.dyn_cast<DistributedTensorType>()) {
        compactType = distributedType.getCompactType();
    } else if (auto sparseType = originalType.dyn_cast<SparseTensorType>()) {
        if (auto distDataType = sparseType.getData().dyn_cast<DistributedTensorType>()) {
            mlir::RankedTensorType dataType = distDataType.getCompactType();
            mlir::RankedTensorType smType = nullptr;
            if (sparseType.getSparsityMap() != nullptr && sparseType.getSparsityMap().isa<DistributedTensorType>()) {
                smType = sparseType.getSparsityMap().cast<DistributedTensorType>().getCompactType();
            }
            mlir::RankedTensorType seType = nullptr;
            if (sparseType.getStorageElementTable() != nullptr &&
                sparseType.getStorageElementTable().isa<DistributedTensorType>()) {
                seType = sparseType.getStorageElementTable().cast<DistributedTensorType>().getCompactType();
            }
            compactType = SparseTensorType::get(dataType, smType, seType, sparseType.getIsWeights(),
                                                sparseType.getSparsityCompression(), sparseType.getSeAttr());
        }
    }
    return compactType;
}

Shape vpux::VPU::getLargestClusterOutputShape(VPU::ClusteredOpInterface clusteredOp,
                                              VPU::MultiClusterStrategy strategy) {
    auto outputType = clusteredOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto numClusters = getOptimalNumClusters(clusteredOp, outputType.getShape(), strategy);
    auto distributedOutputTensorType = getDistributedOutputTypeFromOp(clusteredOp, outputType, numClusters, strategy);
    auto distributedDataType =
            distributedOutputTensorType.getDistributedTypes().front().cast<VPU::DistributedTensorType>();
    return distributedDataType.getLargestCompactShape();
}

bool vpux::VPU::isSegmentedOverlappedAxisSameAsSliceAxis(mlir::ArrayAttr numTiles, ArrayRef<int64_t> inputShape,
                                                         ArrayRef<int64_t> sliceShape) {
    VPUX_THROW_WHEN(numTiles == nullptr, "NumTiles attr is nullptr.");
    const auto numTilesArr = parseIntArrayAttr<int64_t>(numTiles);
    return isSegmentedOverlappedAxisSameAsSliceAxis(numTilesArr, inputShape, sliceShape);
}

bool vpux::VPU::isSegmentedOverlappedAxisSameAsSliceAxis(ArrayRef<int64_t> numTiles, ArrayRef<int64_t> inputShape,
                                                         ArrayRef<int64_t> sliceShape) {
    VPUX_THROW_WHEN(numTiles.empty(), "NumTiles is empty.");
    VPUX_THROW_UNLESS(numTiles.size() == inputShape.size() && numTiles.size() == sliceShape.size(),
                      "NumTiles ({0}), input shape ({1}) and slice shape ({2}) do not have the same number of dims.",
                      numTiles.size(), inputShape.size(), sliceShape.size());

    for (size_t dim = 0; dim < inputShape.size(); dim++) {
        if (inputShape[dim] != sliceShape[dim] && numTiles[dim] != 1) {
            return true;
        }
    }

    return false;
}

SmallVector<int64_t> getFusedKernel(const VPU::DistributionInfo& distTensorType, ArrayRef<int64_t> fusedKernel) {
    if (!fusedKernel.empty()) {
        return SmallVector<int64_t>(fusedKernel);
    }
    const auto kernelAttr = distTensorType.getKernel();
    if (!kernelAttr.empty()) {
        return SmallVector<int64_t>(kernelAttr);
    }
    const auto neutralKernel = SmallVector<int64_t>{1, 1};
    return neutralKernel;
}

SmallVector<int64_t> getFusedStrides(const VPU::DistributionInfo& distTensorType, ArrayRef<int64_t> fusedStrides) {
    if (!fusedStrides.empty()) {
        return SmallVector<int64_t>(fusedStrides);
    }
    const auto stridesAttr = distTensorType.getStrides();
    if (!stridesAttr.empty()) {
        return SmallVector<int64_t>(stridesAttr);
    }
    const auto neutralStrides = SmallVector<int64_t>{1, 1};
    return neutralStrides;
}

VPU::Padding getFusedPads(const VPU::DistributionInfo& distTensorType, const std::optional<Padding>& fusedPads) {
    if (fusedPads.has_value()) {
        return fusedPads.value();
    }
    if (distTensorType.getPadding().has_value()) {
        return distTensorType.getPadding().value();
    }
    return VPU::Padding(0, 0, 0, 0);
}

OverlapDistributionParams getFusedOverlappedParams(const VPU::DistributionInfo& dist,
                                                   const OverlapDistributionParams& fusedOverlapParams,
                                                   const bool equalComputeAndMemoryView) {
    OverlapDistributionParams finalOverlapParams = {};
    if (!fusedOverlapParams.getMemoryShapes().empty() || !dist.getMemoryShapes().empty()) {
        auto finalMemoryShapes = fusedOverlapParams.getMemoryShapes().empty()
                                         ? SmallVector<SmallVector<int64_t>>(dist.getMemoryShapes())
                                         : fusedOverlapParams.getMemoryShapes();
        auto finalMemoryOffsets = fusedOverlapParams.getMemoryOffsets().empty()
                                          ? SmallVector<SmallVector<int64_t>>(dist.getMemoryOffsets())
                                          : fusedOverlapParams.getMemoryOffsets();
        auto finalComputeShapes = fusedOverlapParams.getComputeShapes().empty()
                                          ? SmallVector<SmallVector<int64_t>>(dist.getComputeShapes())
                                          : fusedOverlapParams.getComputeShapes();
        auto finalComputeOffsets = fusedOverlapParams.getComputeOffsets().empty()
                                           ? SmallVector<SmallVector<int64_t>>(dist.getComputeOffsets())
                                           : fusedOverlapParams.getComputeOffsets();

        finalOverlapParams.setMemoryShapes(finalMemoryShapes);
        finalOverlapParams.setMemoryOffsets(finalMemoryOffsets);

        if (equalComputeAndMemoryView) {
            finalOverlapParams.setComputeShapes(finalMemoryShapes);
            finalOverlapParams.setComputeOffsets(finalMemoryOffsets);
        } else {
            finalOverlapParams.setComputeShapes(finalComputeShapes);
            finalOverlapParams.setComputeOffsets(finalComputeOffsets);
        }

        VPUX_THROW_WHEN((finalOverlapParams.getMemoryShapes().empty()) ||
                                (finalOverlapParams.getMemoryOffsets().empty()) ||
                                (finalOverlapParams.getComputeShapes().empty()) ||
                                (finalOverlapParams.getComputeOffsets().empty()),
                        "memoryOffsets/Shapes & computeOffsets/Shapes of finalOverlapParams cannot be nullptr.");

        return finalOverlapParams;
    }

    const auto kernel = getFusedKernel(dist, fusedOverlapParams.getKernel());
    const auto pads = getFusedPads(dist, fusedOverlapParams.getPads());
    const auto strides = getFusedStrides(dist, fusedOverlapParams.getStride());
    finalOverlapParams.setKernel(kernel);
    finalOverlapParams.setPads(pads);
    finalOverlapParams.setStride(strides);
    finalOverlapParams.setEqualComputeAndMemoryView(equalComputeAndMemoryView);
    return finalOverlapParams;
}

VPU::DistributedTensorType vpux::VPU::composeDistributedType(VPU::ClusteredOpInterface permuteOp,
                                                             const VPU::DistributedTensorType distType,
                                                             const vpux::NDTypeInterface ndType,
                                                             const mlir::ArrayAttr tileOverDim,
                                                             const OverlapDistributionParams& fusedOverlapParams,
                                                             const bool enableExplicitDistributionInfoAttr,
                                                             const bool equalComputeAndMemoryView) {
    // Update distributed activation attribute.
    const auto origDistTensorAttr = distType.getDistribution();
    const auto mode = origDistTensorAttr.getMode().getValue();
    const auto numClusters = origDistTensorAttr.getNumClusters().getInt();
    const auto alignment = origDistTensorAttr.getAlignment();
    const auto overlapParams = getFusedOverlappedParams(VPU::DistributionInfo::getClassFromAttr(origDistTensorAttr),
                                                        fusedOverlapParams, equalComputeAndMemoryView);

    const auto tileOverDimArr = tileOverDim ? parseIntArrayAttr<int64_t>(tileOverDim) : SmallVector<int64_t>{};
    const auto alignmentArr = alignment ? parseIntArrayAttr<int64_t>(alignment) : SmallVector<int64_t>{};
    auto uniformDistributedSegments =
            VPU::getUniformDistributedSegments(permuteOp, ndType.getShape().raw(), mode, tileOverDimArr, alignmentArr);

    return createDistributedTensorType(permuteOp, ndType, mode, tileOverDimArr, numClusters, alignmentArr,
                                       uniformDistributedSegments, enableExplicitDistributionInfoAttr, overlapParams);
}

mlir::Operation* vpux::VPU::getNextCompressConv(mlir::Operation* nceOp) {
    if (!nceOp->hasOneUse()) {
        return nullptr;
    }
    mlir::Operation* nextOp = *nceOp->getUsers().begin();
    while (nextOp != nullptr) {
        if (mlir::isa<VPU::ViewLikeOpInterface>(nextOp) && nextOp->hasOneUse()) {
            nextOp = *nextOp->getUsers().begin();
        } else if (mlir::isa<VPU::NCECompressConvolutionOp>(nextOp)) {
            return nextOp;
        } else {
            return nullptr;
        }
    }

    return nullptr;
}

mlir::Type vpux::VPU::fuseOverlapParams(VPU::ClusteredOpInterface permuteOp, const VPU::DistributedTensorType distType,
                                        mlir::Operation* nextConv, bool enableExplicitDistributionInfoAttr) {
    if (nextConv == nullptr) {
        return distType;
    }
    // Get kernel and padding parameters for Permute from trailing convolution.
    VPUX_THROW_UNLESS(mlir::isa<VPU::NCEConvolutionOp>(nextConv) || mlir::isa<VPU::NCECompressConvolutionOp>(nextConv),
                      "Next Conv is neither NCEConv nor NCECompressConv");

    auto conv = mlir::cast<VPU::NCEOpInterface>(nextConv);
    const auto kernel = conv.getKernelSizeVal();
    const auto strides = conv.getStridesVal();
    const auto pads = VPU::Padding::getClassFromAttr(conv.getPad());
    const OverlapDistributionParams overlapParams(kernel, pads, strides, false);

    const auto origDistTensorAttr = distType.getDistribution();
    const auto tileOverDim = origDistTensorAttr.getNumTiles();

    if (auto sparseInputType = distType.dyn_cast<VPU::SparseTensorType>()) {
        const auto dataNdType = sparseInputType.getData().cast<vpux::NDTypeInterface>();
        auto distributedDataType = composeDistributedType(permuteOp, distType, dataNdType, tileOverDim, overlapParams,
                                                          enableExplicitDistributionInfoAttr);
        const auto sparsityNdType = sparseInputType.getSparsityMap().cast<vpux::NDTypeInterface>();
        auto distributedSMType = composeDistributedType(permuteOp, distType, sparsityNdType, tileOverDim, overlapParams,
                                                        enableExplicitDistributionInfoAttr);
        return VPU::SparseTensorType::get(distributedDataType, distributedSMType, nullptr,
                                          sparseInputType.getIsWeights(), sparseInputType.getSparsityCompression());
    }
    const auto ndType = distType.cast<vpux::NDTypeInterface>();
    return composeDistributedType(permuteOp, distType, ndType, tileOverDim, overlapParams,
                                  enableExplicitDistributionInfoAttr);
}

SmallVector<int64_t> vpux::VPU::getNonOneDimInds(ArrayRef<int64_t> inputArray) {
    SmallVector<int64_t> nonOneDims;
    for (auto index : irange(inputArray.size())) {
        if (inputArray[index] != 1) {
            nonOneDims.push_back(checked_cast<int64_t>(index));
        }
    }
    return nonOneDims;
}

mlir::FailureOr<VPU::DistributionInfoAttr> vpux::VPU::legalizeCastedDistribution(
        VPU::DistributionInfoAttr castedDistribution, mlir::MLIRContext* ctx) {
    // Return the original distribution if it's not OVERLAPPED
    if (castedDistribution.getMode().getValue() != VPU::DistributionMode::OVERLAPPED) {
        return castedDistribution;
    }

    const auto numTilesAttr = castedDistribution.getNumTiles();
    // Return the original distribution if no numTilesAttr presents
    if (numTilesAttr == nullptr) {
        return castedDistribution;
    }

    auto numTiles = parseIntArrayAttr<int64_t>(numTilesAttr);
    auto numTileDims = vpux::VPU::getNonOneDimInds(numTiles);
    if (numTileDims.size() != 1) {
        return mlir::failure();
    }

    auto axis = Dim(numTileDims.front());
    // Return the original distribution if it's supported already
    if (axis == Dims4D::Act::W || axis == Dims4D::Act::H) {
        return castedDistribution;
    }

    return VPU::DistributionInfoAttr::get(ctx, VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::SEGMENTED),
                                          castedDistribution.getNumTiles(), nullptr, nullptr, nullptr,
                                          castedDistribution.getNumClusters(), castedDistribution.getAlignment(),
                                          castedDistribution.getUniformDistributedSegments(),
                                          castedDistribution.getComputeShapes(), castedDistribution.getComputeOffsets(),
                                          castedDistribution.getMemoryShapes(), castedDistribution.getMemoryOffsets(),
                                          castedDistribution.getEqualMemoryAndComputeView());
}

mlir::FailureOr<VPU::DistributionInfo> vpux::VPU::legalizeCastedDistribution(
        VPU::DistributionInfo& castedDistribution) {
    // Return the original distribution if it's not OVERLAPPED
    if (castedDistribution.getDistributionMode() != VPU::DistributionMode::OVERLAPPED) {
        return castedDistribution;
    }

    const auto numTiles = castedDistribution.getNumTiles();
    // Return the original distribution if no numTilesAttr presents
    if (numTiles.empty()) {
        return castedDistribution;
    }

    auto numTileDims = vpux::VPU::getNonOneDimInds(numTiles);
    if (numTileDims.size() != 1) {
        return mlir::failure();
    }

    auto axis = Dim(numTileDims.front());
    // Return the original distribution if it's supported already
    if (axis == Dims4D::Act::W || axis == Dims4D::Act::H) {
        return castedDistribution;
    }

    return VPU::DistributionInfo(VPU::DistributionMode::SEGMENTED, castedDistribution.getNumTiles(), {}, {}, {},
                                 castedDistribution.getNumClusters(), castedDistribution.getAlignment(),
                                 castedDistribution.hasUniformDistributedSegments(),
                                 castedDistribution.getComputeShapes(), castedDistribution.getComputeOffsets(),
                                 castedDistribution.getMemoryShapes(), castedDistribution.getMemoryOffsets(),
                                 castedDistribution.hasEqualMemoryAndComputeView());
}

VPU::DistributionInfo vpux::VPU::createDistributionInfo(VPU::SWOpInterface swOp, DistributionMode distributionMode,
                                                        ArrayRef<int64_t> numTiles,
                                                        const int64_t optimalNumberOfClusters,
                                                        ArrayRef<int64_t> alignment,
                                                        const bool uniformDistributedSegments) {
    if (distributionMode == DistributionMode::DUPLICATED) {
        return DistributionInfo(distributionMode, {}, {}, {}, {}, optimalNumberOfClusters, alignment,
                                uniformDistributedSegments, {}, {}, {}, {}, {});
    } else if (VPU::bitEnumContainsAny(distributionMode, VPU::DistributionMode::SEGMENTED)) {
        return DistributionInfo(distributionMode, numTiles, {}, {}, {}, optimalNumberOfClusters, alignment,
                                uniformDistributedSegments, {}, {}, {}, {}, {});
    }

    VPUX_THROW("Unsupported distribution mode: {0} for op {1}", VPU::stringifyDistributionMode(distributionMode), swOp);
    return {};
}

VPU::DistributionInfo vpux::VPU::createDistributionInfo(
        VPU::NCEOpInterface nceOp, DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t optimalNumberOfClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        ArrayRef<int64_t> kernel, const std::optional<VPU::Padding>& pad, ArrayRef<int64_t> stride,
        const bool equalComputeAndMemoryView) {
    if (distributionMode == DistributionMode::OVERLAPPED) {
        return DistributionInfo(distributionMode, numTiles, kernel, stride, pad, optimalNumberOfClusters, alignment,
                                uniformDistributedSegments, {}, {}, {}, {}, equalComputeAndMemoryView);
    } else if (distributionMode == DistributionMode::DUPLICATED) {
        return DistributionInfo(distributionMode, {}, {}, {}, {}, optimalNumberOfClusters, alignment,
                                uniformDistributedSegments, {}, {}, {}, {}, {});
    } else if (VPU ::bitEnumContainsAny(distributionMode, VPU::DistributionMode::SEGMENTED)) {
        return DistributionInfo(distributionMode, numTiles, {}, {}, {}, optimalNumberOfClusters, alignment,
                                uniformDistributedSegments, {}, {}, {}, {}, {});
    }
    VPUX_THROW("Unsupported distribution mode: {0} for op {1}", VPU::stringifyDistributionMode(distributionMode),
               nceOp);
    return {};
}

VPU::DistributionInfo vpux::VPU::createDistributionInfo(
        mlir::Operation* viewLikeOp, DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t optimalNumberOfClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        ArrayRef<int64_t> kernel, const std::optional<VPU::Padding>& pad, ArrayRef<int64_t> stride) {
    VPUX_THROW_UNLESS(mlir::isa_and_nonnull<VPU::ViewLikeOpInterface>(viewLikeOp), "Op {0} is not a view like op",
                      viewLikeOp->getName());

    if (distributionMode == DistributionMode::DUPLICATED) {
        return DistributionInfo(distributionMode, {}, {}, {}, {}, optimalNumberOfClusters, alignment,
                                uniformDistributedSegments, {}, {}, {}, {}, {});
    } else if (VPU ::bitEnumContainsAny(distributionMode, VPU::DistributionMode::SEGMENTED)) {
        return DistributionInfo(distributionMode, numTiles, {}, {}, {}, optimalNumberOfClusters, alignment,
                                uniformDistributedSegments, {}, {}, {}, {}, {});
    } else if (distributionMode == DistributionMode::OVERLAPPED) {
        return DistributionInfo(distributionMode, numTiles, kernel, stride, pad, optimalNumberOfClusters, alignment,
                                uniformDistributedSegments, {}, {}, {}, {}, {});
    }
    VPUX_THROW("Unsupported distribution mode {0} for op {1}", VPU::stringifyDistributionMode(distributionMode),
               viewLikeOp);
    return {};
}

TensorDistributionMap vpux::VPU::getActivationDistributionAttrFromOp(
        VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface inputType, int64_t numClusters,
        SiblingOpsAnalysis& siblingsAnalysis, vpux::NDTypeInterface tiledOutputType, const vpux::TileInfo& tileInfo) {
    VPUX_THROW_UNLESS(clusteredOp.getMultiClusterStrategy().has_value(),
                      "Op {0} does not have multiClusterStrategy attribute", clusteredOp->getLoc());
    return getActivationDistributionAttrFromOp(clusteredOp, inputType, numClusters,
                                               clusteredOp.getMultiClusterStrategy().value(), siblingsAnalysis,
                                               /*customAlignment*/ {}, tiledOutputType, tileInfo);
}

TensorDistributionMap vpux::VPU::getActivationDistributionAttrFromOp(
        VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface inputType, int64_t numClusters,
        VPU::MultiClusterStrategy customStrategy, SiblingOpsAnalysis& siblingsAnalysis,
        ArrayRef<int64_t> customAlignment, vpux::NDTypeInterface tiledOutputType, const vpux::TileInfo& tileInfo) {
    DistributionMode activationTensorDistributionMode;
    SmallVector<int64_t> activationTensorNumTiles;
    if (mlir::isa<VPU::SWOpInterface>(clusteredOp.getOperation())) {
        activationTensorDistributionMode = getSWInputTensorDistributionMode(clusteredOp, customStrategy, inputType);
        activationTensorNumTiles = getSWInputTensorNumTiles(clusteredOp, numClusters, customStrategy, inputType);
    } else {
        activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, customStrategy);
        activationTensorNumTiles = getActivationTensorNumTiles(clusteredOp, numClusters, customStrategy, inputType);
    }

    auto actualOutputType =
            tiledOutputType != nullptr ? tiledOutputType : clusteredOp->getResult(0).getType().cast<NDTypeInterface>();

    auto newCustomAlignment = SmallVector<int64_t>{};
    if (customAlignment.empty()) {
        const auto activationAlignment =
                getActivationTensorAlignment(clusteredOp, numClusters, customStrategy, inputType, actualOutputType);
        if (activationAlignment.has_value()) {
            newCustomAlignment = activationAlignment.value();
        }
    }
    auto uniformDistributedSegments = VPU::getUniformDistributedSegments(clusteredOp, inputType.getShape().raw(),
                                                                         activationTensorDistributionMode,
                                                                         activationTensorNumTiles, newCustomAlignment);

    auto getOverlappedParams = [&]() -> OverlapDistributionParams {
        if (activationTensorDistributionMode != DistributionMode::OVERLAPPED) {
            return OverlapDistributionParams();
        }

        auto swOp = mlir::dyn_cast<VPU::SWOpInterface>(clusteredOp.getOperation());
        if (swOp == nullptr) {
            return getActivationOverlappedParams(clusteredOp, activationTensorNumTiles, uniformDistributedSegments,
                                                 siblingsAnalysis, inputType, tileInfo);
        }

        return getExplicitOverlapParamsForSWOpInput(swOp, actualOutputType.getShape(), activationTensorNumTiles,
                                                    newCustomAlignment);
    };

    const OverlapDistributionParams overlappedParams = getOverlappedParams();

    const auto hasExplicitDistributedAttr = overlappedParams.hasNonnullComputeAndMemoryShapesOffsets();

    if (auto sparseType = inputType.dyn_cast<VPU::SparseTensorType>()) {
        TensorDistributionMap distributions{};
        auto distributedSparseType = createSparseTensorDistributedType(
                clusteredOp, sparseType, activationTensorDistributionMode, activationTensorNumTiles, numClusters,
                newCustomAlignment, uniformDistributedSegments, hasExplicitDistributedAttr, overlappedParams);

        if (auto data = sparseType.getData()) {
            distributions.insert(std::make_pair(
                    data,
                    VPU::DistributionInfo::getClassFromAttr(
                            distributedSparseType.getData().cast<VPU::DistributedTensorType>().getDistribution())));
        }

        if (auto sparsityMap = sparseType.getSparsityMap()) {
            distributions.insert(std::make_pair(
                    sparsityMap, VPU::DistributionInfo::getClassFromAttr(distributedSparseType.getSparsityMap()
                                                                                 .cast<VPU::DistributedTensorType>()
                                                                                 .getDistribution())));
        }

        if (auto seTable = sparseType.getStorageElementTable()) {
            distributions.insert(std::make_pair(
                    seTable, VPU::DistributionInfo::getClassFromAttr(distributedSparseType.getStorageElementTable()
                                                                             .cast<VPU::DistributedTensorType>()
                                                                             .getDistribution())));
        }
        return distributions;
    }

    return TensorDistributionMap{std::make_pair(
            inputType,
            createDistributionInfo(clusteredOp, inputType, activationTensorDistributionMode, activationTensorNumTiles,
                                   numClusters, newCustomAlignment, uniformDistributedSegments,
                                   hasExplicitDistributedAttr, overlappedParams))};
}

TensorDistributionMap vpux::VPU::getOutputDistributionAttrFromOp(VPU::ClusteredOpInterface clusteredOp,
                                                                 vpux::NDTypeInterface outputType, int64_t numClusters,
                                                                 SiblingOpsAnalysis& siblingsAnalysis,
                                                                 ArrayRef<vpux::NDTypeInterface> inputTypes,
                                                                 const vpux::TileInfo& tileInfo,
                                                                 const bool hasExplicitDistributedAttr) {
    VPUX_THROW_UNLESS(clusteredOp.getMultiClusterStrategy().has_value(),
                      "Op {0} does not have multiClusterStrategy attribute", clusteredOp->getLoc());

    return getOutputDistributionAttrFromOp(clusteredOp, outputType, numClusters,
                                           clusteredOp.getMultiClusterStrategy().value(), siblingsAnalysis, inputTypes,
                                           tileInfo, hasExplicitDistributedAttr);
}

TensorDistributionMap vpux::VPU::getOutputDistributionAttrFromOp(VPU::ClusteredOpInterface clusteredOp,
                                                                 vpux::NDTypeInterface outputType, int64_t numClusters,
                                                                 VPU::MultiClusterStrategy customStrategy,
                                                                 SiblingOpsAnalysis& siblingsAnalysis,
                                                                 ArrayRef<vpux::NDTypeInterface> inputTypes,
                                                                 const vpux::TileInfo& tileInfo,
                                                                 const bool hasExplicitDistributedAttr) {
    TensorDistributionMap returnDistributions;
    const auto outputTensorNumTiles =
            vpux::VPU::getOutputTensorNumTiles(clusteredOp, numClusters, customStrategy, outputType);

    // NCEPermute(SOHO) -> Conv(SOHO)
    // The output tensor of the NCEPermute should be OVERLAPPED to avoid spilling
    if (isOverlapOutputPatternRequired(clusteredOp, customStrategy)) {
        const auto origInputType = clusteredOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
        const auto nextConv = getNextCompressConv(clusteredOp.getOperation());
        auto inputDistType = getDistributedActivationTypeFromOp(clusteredOp, origInputType, numClusters, customStrategy)
                                     .cast<VPU::DistributedTensorType>();
        const auto fusedDistType = fuseOverlapParams(clusteredOp, inputDistType, nextConv, hasExplicitDistributedAttr);
        const OverlapDistributionParams permuteOverlapParams = {};
        const auto equalComputeAndMemoryView = true;
        auto distOutAttr =
                composeDistributedAttr(clusteredOp, fusedDistType.cast<VPU::DistributedTensorType>(), outputType,
                                       inputDistType.getDistribution().getNumTiles(), permuteOverlapParams,
                                       hasExplicitDistributedAttr, equalComputeAndMemoryView);

        returnDistributions.insert(std::make_pair(outputType, distOutAttr));
        return returnDistributions;
    }

    const auto outputTensorDistributionMode =
            vpux::VPU::getOutputTensorDistributionMode(clusteredOp, customStrategy, outputType);
    auto outputAlignmentArr = getOutAlignment(clusteredOp, numClusters, customStrategy, inputTypes, outputType);
    auto uniformDistributedSegments =
            VPU::getUniformDistributedSegments(clusteredOp, outputType.getShape().raw(), outputTensorDistributionMode,
                                               outputTensorNumTiles, outputAlignmentArr);

    const OverlapDistributionParams overlappedParams =
            (outputTensorDistributionMode == DistributionMode::OVERLAPPED &&
             !mlir::isa<VPU::SWOpInterface>(clusteredOp.getOperation()))
                    ? getOutputOverlappedParams(clusteredOp, outputTensorNumTiles, uniformDistributedSegments,
                                                outputType, tileInfo, siblingsAnalysis)
                    : OverlapDistributionParams();

    if (auto sparseType = outputType.dyn_cast<VPU::SparseTensorType>()) {
        VPUX_THROW_UNLESS(sparseType.getSparsityMap() != nullptr, "Missing output sparsity map");
        VPUX_THROW_UNLESS(sparseType.getStorageElementTable() == nullptr,
                          "ODU-generated storage element table is not supported");
        auto distributedDataType = createDistributionInfo(
                clusteredOp, sparseType.getData().cast<vpux::NDTypeInterface>(), outputTensorDistributionMode,
                outputTensorNumTiles, numClusters, outputAlignmentArr, uniformDistributedSegments,
                hasExplicitDistributedAttr, overlappedParams);
        auto distributedSMType = createDistributionInfo(
                clusteredOp, sparseType.getSparsityMap().cast<vpux::NDTypeInterface>(), outputTensorDistributionMode,
                outputTensorNumTiles, numClusters, outputAlignmentArr, uniformDistributedSegments,
                hasExplicitDistributedAttr, overlappedParams);

        returnDistributions.insert(
                std::make_pair(sparseType.getData().cast<vpux::NDTypeInterface>(), distributedDataType));
        returnDistributions.insert(
                std::make_pair(sparseType.getSparsityMap().cast<vpux::NDTypeInterface>(), distributedSMType));

        return returnDistributions;
    }

    auto distribution = createDistributionInfo(
            clusteredOp, outputType, outputTensorDistributionMode, outputTensorNumTiles, numClusters,
            outputAlignmentArr, uniformDistributedSegments, hasExplicitDistributedAttr, overlappedParams);
    returnDistributions.insert(std::make_pair(outputType, distribution));
    return returnDistributions;
}

TensorDistributionMap vpux::VPU::getFilterDistributionAttrFromOp(VPU::NCEOpInterface nceOp,
                                                                 vpux::NDTypeInterface inputType, int64_t numClusters,
                                                                 VPU::MultiClusterStrategy customStrategy) {
    TensorDistributionMap returnDistributions;
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    const auto weightsTensorDistributionMode = getWeightsTensorDistributionMode(customStrategy);
    const auto weightsTensorNumTiles = getWeightsTensorNumTiles(clusteredOp, inputType, numClusters, customStrategy);

    const auto weightAlignment = getWeightsTensorAlignment(customStrategy);

    const auto weightAlignmentArr = weightAlignment.has_value() ? weightAlignment.value() : SmallVector<int64_t>{};

    auto uniformDistributedSegments = VPU::getUniformDistributedSegments(
            mlir::cast<VPU::ClusteredOpInterface>(nceOp.getOperation()), inputType.getShape().raw(),
            weightsTensorDistributionMode, weightsTensorNumTiles, weightAlignmentArr);

    if (auto sparseType = inputType.dyn_cast<VPU::SparseTensorType>()) {
        VPUX_THROW_UNLESS(sparseType.getSparsityMap() != nullptr, "Missing filter sparsity map");
        VPUX_THROW_UNLESS(sparseType.getStorageElementTable() == nullptr,
                          "Storage element table is not supported for weights input");
        auto distributedDataType = VPU::DistributionInfo(weightsTensorDistributionMode, weightsTensorNumTiles, {}, {},
                                                         Padding(), numClusters, weightAlignmentArr,
                                                         uniformDistributedSegments, {}, {}, {}, {}, false);
        auto distributedSMType = VPU::DistributionInfo(weightsTensorDistributionMode, weightsTensorNumTiles, {}, {},
                                                       Padding(), numClusters, weightAlignmentArr,
                                                       uniformDistributedSegments, {}, {}, {}, {}, false);

        returnDistributions.insert(
                std::make_pair(sparseType.getData().cast<vpux::NDTypeInterface>(), distributedDataType));
        returnDistributions.insert(
                std::make_pair(sparseType.getSparsityMap().cast<vpux::NDTypeInterface>(), distributedSMType));

        return returnDistributions;
    }

    auto distribution =
            VPU::DistributionInfo(weightsTensorDistributionMode, weightsTensorNumTiles, {}, {}, Padding(), numClusters,
                                  weightAlignmentArr, uniformDistributedSegments, {}, {}, {}, {}, false);
    returnDistributions.insert(std::make_pair(inputType, distribution));
    return returnDistributions;
}

VPU::DistributionInfo vpux::VPU::createDistributionInfo(VPU::ClusteredOpInterface clusteredOp,
                                                        vpux::NDTypeInterface inputType,
                                                        DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
                                                        const int64_t numClusters, ArrayRef<int64_t> alignment,
                                                        const bool uniformDistributedSegments,
                                                        const bool hasExplicitDistributionInfoAttribute,
                                                        const VPU::OverlapDistributionParams& overlapParams) {
    if (hasExplicitDistributionInfoAttribute || overlapParams.hasNonnullComputeAndMemoryShapesOffsets()) {
        numTiles = (VPU::bitEnumContainsAny(distributionMode, DistributionMode::OVERLAPPED) ||
                    VPU::bitEnumContainsAny(distributionMode, DistributionMode::SEGMENTED))
                           ? numTiles
                           : SmallVector<int64_t>{};
        return clusteredOp.getExplicitDistributionInfoAttr(inputType.getShape(), distributionMode, numTiles,
                                                           numClusters, alignment, uniformDistributedSegments,
                                                           overlapParams);
    }

    return llvm::TypeSwitch<mlir::Operation*, DistributionInfo>(clusteredOp.getOperation())
            .Case<VPU::SWOpInterface>([&](VPU::SWOpInterface swOp) {
                return createDistributionInfo(swOp, distributionMode, numTiles, numClusters, alignment,
                                              uniformDistributedSegments);
            })
            .Case<VPU::NCEOpInterface>([&](VPU::NCEOpInterface nceOp) {
                return createDistributionInfo(nceOp, distributionMode, numTiles, numClusters, alignment,
                                              uniformDistributedSegments, overlapParams.getKernel(),
                                              overlapParams.getPads(), overlapParams.getStride(),
                                              overlapParams.hasEqualComputeAndMemoryView());
            })
            .Case<VPU::ConcatOp>([&](VPU::ConcatOp concatOp) {
                return createDistributionInfo(concatOp.getOperation(), distributionMode, numTiles, numClusters,
                                              alignment, uniformDistributedSegments, overlapParams.getKernel(),
                                              overlapParams.getPads(), overlapParams.getStride());
            })
            .Default([clusteredOp](mlir::Operation*) -> DistributionInfo {
                VPUX_THROW("unsupported operation for createDistributedTensor: {0}", clusteredOp);
            });
}

VPU::DistributionInfo vpux::VPU::composeDistributedAttr(VPU::ClusteredOpInterface permuteOp,
                                                        const VPU::DistributedTensorType distType,
                                                        const vpux::NDTypeInterface ndType,
                                                        const mlir::ArrayAttr tileOverDim,
                                                        const OverlapDistributionParams& fusedOverlapParams,
                                                        const bool enableExplicitDistributedTensor,
                                                        const bool equalComputeAndMemoryView) {
    // Update distributed activation attribute.
    const auto origDistTensorAttr = distType.getDistribution();
    const auto mode = origDistTensorAttr.getMode().getValue();
    const auto numClusters = origDistTensorAttr.getNumClusters().getInt();
    const auto alignment = origDistTensorAttr.getAlignment();

    const auto overlapParams = getFusedOverlappedParams(VPU::DistributionInfo::getClassFromAttr(origDistTensorAttr),
                                                        fusedOverlapParams, equalComputeAndMemoryView);

    const auto tileOverDimArr = tileOverDim ? parseIntArrayAttr<int64_t>(tileOverDim) : SmallVector<int64_t>{};
    const auto alignmentArr = alignment ? parseIntArrayAttr<int64_t>(alignment) : SmallVector<int64_t>{};
    auto uniformDistributedSegments =
            VPU::getUniformDistributedSegments(permuteOp, ndType.getShape().raw(), mode, tileOverDimArr, alignmentArr);

    return createDistributionInfo(permuteOp, ndType, mode, tileOverDimArr, numClusters, alignmentArr,
                                  uniformDistributedSegments, enableExplicitDistributedTensor, overlapParams);
}

vpux::Byte vpux::VPU::getTotalAllocSizeWithDistribution(vpux::NDTypeInterface type,
                                                        const VPU::DistributionInfo& distribution) {
    SmallVector<Shape> perClusterShapes{};
    if (distribution.getMemoryShapes().size() == 0) {
        auto optionalPerClusterMemoryShapes = VPU::getPerClusterMemoryShapes(type.getShape(), distribution);
        VPUX_THROW_UNLESS(optionalPerClusterMemoryShapes.has_value(),
                          "Cannot get per cluster memory shapes. Shape {0}, Unsupported distribution: {1}",
                          type.getShape(), distribution);
        perClusterShapes = optionalPerClusterMemoryShapes.value();
    } else {
        for (auto& shape : distribution.getMemoryShapes()) {
            perClusterShapes.push_back(Shape(shape));
        }
    }
    const Shape tiledShape =
            *std::max_element(perClusterShapes.begin(), perClusterShapes.end(), [](ShapeRef a, ShapeRef b) {
                return vpux::details::calcTotalShapeSize(a.raw()) < vpux::details::calcTotalShapeSize(b.raw());
            });

    const auto totalSize = vpux::details::calcTotalShapeSize(tiledShape.raw());
    const Bit elemSize = type.getElemTypeSize();

    return alignMemSize(elemSize * totalSize, Byte(1)).to<Byte>();
}

vpux::Byte vpux::VPU::getTotalAllocSizeWithDistribution(vpux::NDTypeInterface type,
                                                        const TensorDistributionMap& distributions) {
    Byte totalSize(0);
    if (auto sparseTensor = mlir::dyn_cast<VPU::SparseTensorType>(type)) {
        if (auto sparsityCompression = sparseTensor.getSparsityCompression()) {
            totalSize += sparsityCompression.getAllocSize(sparseTensor.getElementType());
        } else {
            const auto data = sparseTensor.getData().cast<NDTypeInterface>();
            totalSize += getTotalAllocSizeWithDistribution(data, distributions.at(data));
        }
        if (auto sparsityMap = sparseTensor.getSparsityMap()) {
            totalSize += getTotalAllocSizeWithDistribution(sparsityMap, distributions.at(sparsityMap));
        }
        if (auto SETable = sparseTensor.getStorageElementTable()) {
            totalSize += getTotalAllocSizeWithDistribution(SETable, distributions.at(SETable));
        }
    } else {
        totalSize += getTotalAllocSizeWithDistribution(type, distributions.at(type));
    }

    return totalSize;
}

vpux::NDTypeInterface vpux::VPU::getDistributedTypeFromDistributionMap(vpux::NDTypeInterface type,
                                                                       const TensorDistributionMap& distributionMap) {
    auto ctx = type.getContext();

    const auto memSpace = vpux::IndexedSymbolAttr::get(ctx, stringifyEnum(MemoryKind::CMX_NN));
    const auto order = mlir::AffineMapAttr::get(type.getDimsOrder().toAffineMap(ctx));
    auto elemType = type.getElementType();

    if (auto sparseTensor = mlir::dyn_cast<VPU::SparseTensorType>(type)) {
        auto dataType = sparseTensor.getData().cast<NDTypeInterface>();
        vpux::NDTypeInterface distributedDataType = dataType;
        if (distributionMap.contains(dataType)) {
            auto distribution = distributionMap.at(dataType);
            if (distribution.getDistributionMode() != DistributionMode::NONE) {
                const auto order = mlir::AffineMapAttr::get(dataType.getDimsOrder().toAffineMap(ctx));
                distributedDataType = VPU::DistributedTensorType::get(
                        ctx, dataType.getShape().raw(), dataType.getElementType(), order, memSpace,
                        VPU::DistributionInfo::getAttrFromClass(ctx, distribution));
            }
        }
        vpux::NDTypeInterface distributedSMType = nullptr;
        if (auto smType = mlir::dyn_cast_or_null<NDTypeInterface>(sparseTensor.getSparsityMap())) {
            distributedSMType = smType;
            if (distributionMap.contains(smType)) {
                auto distribution = distributionMap.at(smType);
                if (distribution.getDistributionMode() != DistributionMode::NONE) {
                    const auto order = mlir::AffineMapAttr::get(smType.getDimsOrder().toAffineMap(ctx));
                    distributedSMType = VPU::DistributedTensorType::get(
                            ctx, smType.getShape().raw(), smType.getElementType(), order, memSpace,
                            VPU::DistributionInfo::getAttrFromClass(ctx, distribution));
                }
            }
        }
        vpux::NDTypeInterface distributedSEType = nullptr;
        if (auto seType = mlir::dyn_cast_or_null<NDTypeInterface>(sparseTensor.getStorageElementTable())) {
            distributedSEType = seType;
            if (distributionMap.contains(seType)) {
                auto distribution = distributionMap.at(seType);
                if (distribution.getDistributionMode() != DistributionMode::NONE) {
                    const auto order = mlir::AffineMapAttr::get(seType.getDimsOrder().toAffineMap(ctx));
                    distributedSEType = VPU::DistributedTensorType::get(
                            ctx, seType.getShape().raw(), seType.getElementType(), order, memSpace,
                            VPU::DistributionInfo::getAttrFromClass(ctx, distribution));
                }
            }
        }
        return VPU::SparseTensorType::get(distributedDataType, distributedSMType, distributedSEType,
                                          sparseTensor.getIsWeights(), sparseTensor.getSparsityCompression(),
                                          sparseTensor.getSeAttr());
    }

    if (distributionMap.contains(type)) {
        auto distribution = distributionMap.at(type);
        if (distribution.getDistributionMode() == DistributionMode::NONE) {
            return type;
        }
        return VPU::DistributedTensorType::get(ctx, type.getShape().raw(), elemType, order, memSpace,
                                               VPU::DistributionInfo::getAttrFromClass(ctx, distribution));
    }
    return type;
}

TensorDistributionMap vpux::VPU::getDistributionMapFromDistributedType(vpux::NDTypeInterface type) {
    TensorDistributionMap distributionMap;
    if (auto sparseTensor = mlir::dyn_cast<VPU::SparseTensorType>(type)) {
        auto dataType = mlir::cast<NDTypeInterface>(sparseTensor.getData());
        if (auto distributedDataType = mlir::dyn_cast<VPU::DistributedTensorType>(dataType)) {
            auto distribution = VPU::DistributionInfo::getClassFromAttr(distributedDataType.getDistribution());
            distributionMap.insert(std::make_pair(dataType, distribution));
        } else {
            distributionMap.insert(std::make_pair(dataType, VPU::DistributionInfo{}));
        }
        if (auto smType = mlir::dyn_cast_or_null<NDTypeInterface>(sparseTensor.getSparsityMap())) {
            if (auto distributedSMType = mlir::dyn_cast<VPU::DistributedTensorType>(smType)) {
                auto distribution = VPU::DistributionInfo::getClassFromAttr(distributedSMType.getDistribution());
                distributionMap.insert(std::make_pair(smType, distribution));
            } else {
                distributionMap.insert(std::make_pair(smType, VPU::DistributionInfo{}));
            }
        }
        if (auto seType = mlir::dyn_cast_or_null<NDTypeInterface>(sparseTensor.getStorageElementTable())) {
            if (auto distributedSEType = mlir::dyn_cast<VPU::DistributedTensorType>(seType)) {
                auto distribution = VPU::DistributionInfo::getClassFromAttr(distributedSEType.getDistribution());
                distributionMap.insert(std::make_pair(seType, distribution));
            } else {
                distributionMap.insert(std::make_pair(seType, VPU::DistributionInfo{}));
            }
        }
    } else {
        if (auto distributedType = mlir::dyn_cast<VPU::DistributedTensorType>(type)) {
            auto distribution = VPU::DistributionInfo::getClassFromAttr(distributedType.getDistribution());
            distributionMap.insert(std::make_pair(distributedType, distribution));
        } else {
            distributionMap.insert(std::make_pair(type, VPU::DistributionInfo{}));
        }
    }
    return distributionMap;
}

bool vpux::VPU::isSegmentedLikeDistributionMode(vpux::NDTypeInterface sourceType,
                                                const VPU::DistributionInfo& sourceDistribution) {
    if (sourceType == nullptr || sourceDistribution.getDistributionMode() == DistributionMode::NONE) {
        return false;
    }
    const auto distributionMode = sourceDistribution.getDistributionMode();
    if (distributionMode == DistributionMode::SEGMENTED) {
        return true;
    }
    if (distributionMode != DistributionMode::OVERLAPPED) {
        return false;
    }
    // Check if OVERLAPPED per cluster shapes and offsets are identical to the SEGMENTED mode
    if (sourceDistribution.getNumTiles().empty()) {
        return false;
    }

    const auto segmentedDistribution = getNonOverlappedDistributedNative(
            sourceType.getShape(), VPU::DistributionMode::SEGMENTED, sourceDistribution.getNumTiles(),
            sourceDistribution.getNumClusters(), sourceDistribution.getAlignment(),
            sourceDistribution.hasUniformDistributedSegments());
    return arePerClusterMemoryShapeAndOffsetsEqual(sourceType, sourceDistribution, segmentedDistribution);
}
