//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/overlap_distribution_utils.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"

using namespace vpux;
using namespace VPU;

namespace {
SmallVector<Shape> getPerClusterEndOffset(ArrayRef<Shape> startOffset, ArrayRef<Shape> size) {
    const auto numDims = startOffset[0].size();
    const auto numClusters = startOffset.size();
    auto endOffset = SmallVector<Shape>(numClusters, Shape(numDims, 0));

    for (size_t cluster = 0; cluster < numClusters; cluster++) {
        for (size_t dim = 0; dim < numDims; dim++) {
            endOffset[cluster][Dim(dim)] = startOffset[cluster][Dim(dim)] + size[cluster][Dim(dim)] - 1;
        }
    }

    return endOffset;
}

bool isDistributedTensorSOH(VPU::DistributedTensorType distributedTensorType) {
    auto distribution = distributedTensorType.getDistribution();
    const auto mode = distribution.getMode().getValue();

    const bool isSegOverlappedMode =
            mode == VPU::DistributionMode::SEGMENTED || mode == VPU::DistributionMode::OVERLAPPED;
    if (!isSegOverlappedMode) {
        return false;
    }

    const auto numTiles = parseIntArrayAttr<int64_t>(distribution.getNumTiles());
    return numTiles[Dims4D::Act::H.ind()] != 1;
}

bool isValidCandidateForCMXConcat(mlir::Operation* maybeConcat) {
    auto concat = mlir::dyn_cast_or_null<VPU::ConcatOp>(maybeConcat);
    if (concat == nullptr) {
        return false;
    }

    bool isProducerOrConsumerSOH = false;
    for (const auto& producerConcat : concat->getOperands()) {
        if (!mlir::isa_and_nonnull<VPU::NCEOpInterface, VPU::NCEClusterTilingOp>(producerConcat.getDefiningOp())) {
            return false;
        }

        if (auto clusterTilingCopy = mlir::dyn_cast_or_null<VPU::NCEClusterTilingOp>(producerConcat.getDefiningOp())) {
            auto copyOp = clusterTilingCopy.getInnerTaskOpOfType<VPU::CopyOp>();
            if (!copyOp) {
                return false;
            }
            auto clusterTilingNCE = clusterTilingCopy->getOperand(0).getDefiningOp<VPU::NCEClusterTilingOp>();
            if (!clusterTilingNCE) {
                return false;
            }
            auto nceOp = clusterTilingNCE.getInnerTaskOpOfType<VPU::NCEOpInterface>();
            if (!nceOp) {
                return false;
            }

            auto distributedTensorType = clusterTilingNCE.getResult(0).getType().dyn_cast<VPU::DistributedTensorType>();
            if (distributedTensorType == nullptr) {
                return false;
            }

            isProducerOrConsumerSOH = isDistributedTensorSOH(distributedTensorType);
        } else if (auto nceOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(producerConcat.getDefiningOp())) {
            if (nceOp.getMultiClusterStrategy().has_value()) {
                if (nceOp.getMultiClusterStrategy().value() == VPU::MultiClusterStrategy::SplitOverHeight) {
                    isProducerOrConsumerSOH = true;
                }
            }
        }
    }
    for (const auto& consumerConcat : concat->getUsers()) {
        if (mlir::isa_and_nonnull<mlir::func::ReturnOp>(consumerConcat)) {
            continue;
        }

        if (!mlir::isa_and_nonnull<VPU::NCEOpInterface, VPU::NCEClusterTilingOp>(consumerConcat)) {
            return false;
        }

        if (auto clusterTilingCopy = mlir::dyn_cast<VPU::NCEClusterTilingOp>(consumerConcat)) {
            auto copyOp = clusterTilingCopy.getInnerTaskOpOfType<VPU::CopyOp>();
            if (!copyOp) {
                return false;
            }
            if (!clusterTilingCopy->hasOneUse()) {
                return false;
            }
            auto clusterTilingNCE = mlir::dyn_cast_or_null<VPU::NCEClusterTilingOp>(*clusterTilingCopy->user_begin());
            if (!clusterTilingNCE) {
                return false;
            }
            auto nceOp = clusterTilingNCE.getInnerTaskOpOfType<VPU::NCEOpInterface>();
            if (!nceOp) {
                return false;
            }

            auto distributedTensorType =
                    clusterTilingCopy.getResult(0).getType().dyn_cast<VPU::DistributedTensorType>();
            if (distributedTensorType == nullptr) {
                return false;
            }

            isProducerOrConsumerSOH = isDistributedTensorSOH(distributedTensorType);
        } else if (auto nceOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(consumerConcat)) {
            if (nceOp.getMultiClusterStrategy().has_value()) {
                if (nceOp.getMultiClusterStrategy().value() == VPU::MultiClusterStrategy::SplitOverHeight) {
                    isProducerOrConsumerSOH = true;
                }
            }
        }
    }

    auto isOffsetOnH = [](mlir::ArrayAttr offset) {
        auto offsetVector = Shape(parseIntArrayAttr<int64_t>(offset));
        return offsetVector[Dims4D::Act::H] != 0;
    };

    bool isConcatOverH = false;
    if (concat.getStaticOffsets().has_value()) {
        const auto concatDims = concat.getStaticOffsetsAttr().getAsRange<mlir::ArrayAttr>();
        isConcatOverH = llvm::any_of(concatDims, isOffsetOnH);
    } else if (concat.getPerAxis().has_value()) {
        const auto concatAxis = concat.getPerAxis().value().getAxis().getValue().getSExtValue();
        isConcatOverH = concatAxis == Dims4D::Act::H.ind();
    }

    return !(isProducerOrConsumerSOH && isConcatOverH);
}

int64_t extractKernelTileAxis(ArrayRef<int64_t> numTiles) {
    VPUX_THROW_UNLESS(numTiles[Dims4D::Act::H.ind()] == 1 || numTiles[Dims4D::Act::W.ind()] == 1,
                      "Multidimension cluster tiling across H and W is not yet supported.");
    if (numTiles[Dims4D::Act::W.ind()] > 1) {
        return Dims4D::Kernel::X.ind();
    }
    return Dims4D::Kernel::Y.ind();
}

// Currently supported "passthrough" ops where Overlapped params are concerned
// Concat not on clustering dim, Eltwise with in-place attr and QuantizeCast act as sort of "passthrough" ops.
// For example, given the following subgraph:
// PASS_OP is (Eltwise in place) || (valid CMXConcat) || QuantizeCast
//
//     DPU0           DPU1
//    /    \         /   |
// DPU3      PASS_OP    DPU4
//          /     |
//      DPU2     DPU5
//
// Sibling ops will be DPU2, DPU3, DPU4 and DPU5 i.e. the siblings of the PASS_OP & its consumers

bool isBranchingPassthroughOp(mlir::Operation* op) {
    if (isValidCandidateForCMXConcat(op)) {
        return true;
    }

    if (auto eltwise = mlir::dyn_cast_or_null<VPU::NCEEltwiseOp>(op)) {
        if (eltwise.getIsInplace().value_or(false)) {
            return true;
        }
    }

    return false;
}

bool isPassthroughOp(mlir::Operation* op) {
    // Given the following subgraph:
    //             Producer
    //         /             |
    // [QuantizeCast]  [QuantizeCast]
    //       |               |
    //     NceOp0          NceOp1
    // NceOp0 and NceOp1 should be aware of each other as siblings to be able to properly set their input distributions
    // TODO: 104112 avoid spilling due to other view ops besides of QuantizeCast
    if (mlir::isa<VPU::QuantizeCastOp>(op) || mlir::isa<VPU::GroupSparseTensorOp>(op)) {
        return true;
    }

    return isBranchingPassthroughOp(op);
}

void findSiblings(mlir::Value operand, std::set<VPU::ClusteredOpInterface>& siblings,
                  std::set<llvm::hash_code>& visited, const int64_t level = 0) {
    const auto operandHash = mlir::hash_value(operand);
    const bool tensorIsVisited = visited.count(operandHash) == 1;
    if (tensorIsVisited) {
        return;
    }

    visited.emplace(operandHash);

    auto findSiblingsForPassthroughOrMultiInputOp = [&](mlir::Operation* user) {
        if (auto clusteredUser = mlir::dyn_cast<VPU::ClusteredOpInterface>(user)) {
            // if op is a sibling already, it's consumers/producers were already iterated through
            if (siblings.count(clusteredUser) != 0) {
                return;
            }

            siblings.emplace(clusteredUser);
        }

        // In scenarios where there are a lot of intertwined passtrough ops, compilation
        // time increases by large margin. We're using the level to prevent the neighbour
        // searching algo from going through too many ops.
        // TODO: Better handling E#115755
        if (level < 0) {
            return;
        }

        if (isPassthroughOp(user) && user->getResult(0) != operand) {
            findSiblings(user->getResult(0), siblings, visited, level - 1);
        }

        // For subgraph such as:
        //             Producer
        //         /             |
        // [QuantizeCast]  [QuantizeCast]
        //       |               |
        //     NceOp0          NceOp1
        // Decreasing the level will lead to the the 2 NCE ops no longer being
        // discovered as siblings; prevent that by trying to manually look for another
        // QuantizeCast/GroupSparseBuffer sibling.
        // TODO: Better handling E#115755
        const bool notBranchingPassthrough = !isBranchingPassthroughOp(user) && isPassthroughOp(user);
        for (const auto siblingOperand : user->getOperands()) {
            if (siblingOperand != operand) {
                if (notBranchingPassthrough) {
                    for (const auto siblingUser : siblingOperand.getUsers()) {
                        if (!isBranchingPassthroughOp(siblingUser) && isPassthroughOp(siblingUser)) {
                            findSiblings(siblingUser->getResult(0), siblings, visited, level - 1);
                        }
                    }
                }
                findSiblings(siblingOperand, siblings, visited, level - 1);
            }
        }
    };

    if (operand.getDefiningOp() != nullptr && isPassthroughOp(operand.getDefiningOp())) {
        findSiblingsForPassthroughOrMultiInputOp(operand.getDefiningOp());
    }

    for (const auto& sibling : operand.getUsers()) {
        if (isPassthroughOp(sibling)) {
            findSiblingsForPassthroughOrMultiInputOp(sibling);
        } else if (mlir::isa_and_nonnull<VPU::NCEEltwiseOp>(sibling)) {
            findSiblingsForPassthroughOrMultiInputOp(sibling);
        } else if (auto clusteredUser = mlir::dyn_cast<VPU::ClusteredOpInterface>(sibling)) {
            siblings.emplace(clusteredUser);
        }
    }
}

}  // namespace

OverlapDistributionParams vpux::VPU::getOverlappedDistributionParameters(mlir::MLIRContext* ctx,
                                                                         ArrayRef<VPU::ClusteredOpInterface> opSubgraph,
                                                                         int64_t kernelDistributionAxis,
                                                                         mlir::UnitAttr equalComputeAndMemoryView) {
    auto kernel = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    auto pads = VPU::getPaddingAttr(ctx, 0, 0, 0, 0);
    auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});

    SmallVector<VPU::NCEOpInterface> nceOpCandidates;
    for (auto clusteredOp : opSubgraph) {
        // clusteredOp with SOHO strategy satisfy below SOH condition too so won't be dropped
        if (clusteredOp.isOperationSplitOverHeightCompatible(/*vpux::TileInfo=*/vpux::TileInfo(ShapeRef()))) {
            if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(clusteredOp.getOperation())) {
                nceOpCandidates.push_back(nceOp);
            }
        }
    }

    if (nceOpCandidates.empty()) {
        return OverlapDistributionParams(kernel, pads, strides);
    }

    // For now just take the highest kernel
    // As we have better representation in distributedBuffer, switch to computing the
    // actual shapes per clusters

    auto largestKernel = 0;
    auto largestIndex = 0;
    for (auto it : nceOpCandidates | indexed) {
        auto kernelSize = it.value().getKernelSizeVal()[kernelDistributionAxis];
        if (kernelSize > largestKernel) {
            largestKernel = kernelSize;
            largestIndex = it.index();
        }
    }

    kernel = getIntArrayAttr(ctx, nceOpCandidates[largestIndex].getKernelSizeVal());
    pads = nceOpCandidates[largestIndex].getPad();
    strides = getIntArrayAttr(ctx, nceOpCandidates[largestIndex].getStridesVal());

    return OverlapDistributionParams(kernel, pads, strides, equalComputeAndMemoryView);
}

// Version with extension of halo
OverlapDistributionParams vpux::VPU::getOverlappedDistributionParameters(
        mlir::MLIRContext* ctx, NDTypeInterface tensorType, ArrayRef<VPU::ClusteredOpInterface> consumerSubgraph,
        const int64_t numClusters, ArrayRef<int64_t> numTiles, mlir::UnitAttr uniformDistributedSegments,
        const vpux::TileInfo& tileInfo) {
    VPUX_THROW_WHEN(tensorType == nullptr, "getOverlappedDistributionParameters: tensorType cannot be nullptr");

    auto numClustersAttr = getIntAttr(ctx, numClusters);
    auto neutralKernel = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    auto neutralPads = VPU::getPaddingAttr(ctx, 0, 0, 0, 0);
    auto neutralStrides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});

    auto numTilesAttr = getIntArrayAttr(ctx, numTiles);

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::OVERLAPPED);
    auto neutralDistributedAttr = VPU::DistributedTensorAttr::get(
            ctx, distributionModeAttr, numTilesAttr, neutralKernel, neutralPads, neutralStrides, numClustersAttr,
            nullptr, uniformDistributedSegments, nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto shape = tensorType.getShape();
    const auto computeShapes = getPerClusterComputeShapes(shape, neutralDistributedAttr);
    const auto computeOffsets = getPerClusterComputeShapeOffsets(shape, neutralDistributedAttr);
    auto memoryShapes = computeShapes;
    auto memoryOffsets = computeOffsets;

    SmallVector<VPU::NCEOpInterface> nceOpCandidates;
    for (auto clusteredOp : consumerSubgraph) {
        if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(clusteredOp.getOperation())) {
            nceOpCandidates.push_back(nceOp);
        }
    }

    if (nceOpCandidates.empty()) {
        return OverlapDistributionParams(
                vpux::getIntArrayOfArray(ctx, memoryShapes), vpux::getIntArrayOfArray(ctx, memoryOffsets),
                vpux::getIntArrayOfArray(ctx, computeShapes), vpux::getIntArrayOfArray(ctx, computeOffsets));
    }

    const auto kernelTileAxis = extractKernelTileAxis(numTiles);
    const auto clusteringAxis = VPU::getDistributedTilingAxis(numTiles);
    const auto tileOffset = !tileInfo.offsets.empty() ? tileInfo.offsets[Dim(clusteringAxis)] : 0;

    // TODO: E#112803 Add support for extended memory view for sparse types with SETable; remove throw after
    // implementation is done.
    const auto sparseTensor = tensorType.dyn_cast<VPU::SparseTensorType>();
    const bool origTensorHasSETable = (sparseTensor != nullptr) && (sparseTensor.getSeAttr() != nullptr);
    VPUX_THROW_WHEN(origTensorHasSETable && nceOpCandidates.size() != 1,
                    "Equalizing memory view between a SEP op and another NCE op is not supported, yet.");

    for (auto nceOp : nceOpCandidates) {
        auto kernelSize = getIntArrayAttr(ctx, nceOp.getKernelSizeVal());
        auto stridesSize = getIntArrayAttr(ctx, nceOp.getStridesVal());
        auto padsSize = nceOp.getPad();

        // WA to get correct pad start when in a tiling scenario
        // TODO: Proper fix - E#112801
        if (tileOffset != 0) {
            if (kernelTileAxis == Dims4D::Kernel::Y.ind()) {
                padsSize = VPU::getPaddingAttr(ctx, padsSize.getLeft().getInt(), padsSize.getRight().getInt(), 0,
                                               padsSize.getBottom().getInt());
            } else {
                padsSize = VPU::getPaddingAttr(ctx, 0, padsSize.getRight().getInt(), padsSize.getTop().getInt(),
                                               padsSize.getBottom().getInt());
            }
        }

        auto consumerDistr = VPU::DistributedTensorAttr::get(
                ctx, distributionModeAttr, numTilesAttr, kernelSize, padsSize, stridesSize, numClustersAttr, nullptr,
                uniformDistributedSegments, nullptr, nullptr, nullptr, nullptr, nullptr);

        auto tensorShape = shape;

        const auto sparseNceOpInputType = nceOp->getOperand(0).getType().dyn_cast<VPU::SparseTensorType>();
        const bool currentNceOpHasSETable =
                sparseNceOpInputType != nullptr && sparseNceOpInputType.getSeAttr() != nullptr;

        if (currentNceOpHasSETable) {
            if (nceOpCandidates.size() != 1) {
                // ignore consumers with SETable if there are other consumers
                continue;
            }

            // we decide memory view for producer op with only SEPOp as consumer
            if (!origTensorHasSETable) {
                tensorShape = sparseNceOpInputType.getSeAttr().inferOutputShape(shape);
            }
        }

        auto consumerMemoryShapesOpt = getPerClusterMemoryShapes(tensorShape, consumerDistr);
        if (!consumerMemoryShapesOpt.has_value()) {
            // op is not SOH/SOW Compatible
            continue;
        }

        auto consumerMemoryShapes = consumerMemoryShapesOpt.value();
        auto consumerMemoryOffsets = getPerClusterMemoryShapeOffsets(tensorShape, consumerDistr);

        if (origTensorHasSETable) {
            memoryShapes = consumerMemoryShapes;
            memoryOffsets = consumerMemoryOffsets;
            continue;
        }

        const bool opHasConsumerWithSETable = currentNceOpHasSETable && !origTensorHasSETable;
        // E#112803 support for SEP should be added
        if (opHasConsumerWithSETable) {
            continue;
        }
        for (int64_t cluster = 0; cluster < numClusters; ++cluster) {
            auto& crtClusterMemoryOffsets = memoryOffsets[cluster];
            auto& crtClusterMemoryShapes = memoryShapes[cluster];

            auto crtClusterConsMemOffsets = consumerMemoryOffsets[cluster];
            auto crtClusterConsMemShapes = consumerMemoryShapes[cluster];

            auto endOffset =
                    crtClusterMemoryOffsets[Dim(clusteringAxis)] + crtClusterMemoryShapes[Dim(clusteringAxis)] - 1;
            const auto candidateEndOffset =
                    crtClusterConsMemOffsets[Dim(clusteringAxis)] + crtClusterConsMemShapes[Dim(clusteringAxis)] - 1;

            auto startOffset = crtClusterMemoryOffsets[Dim(clusteringAxis)];
            const auto candidateStartOffset = crtClusterConsMemOffsets[Dim(clusteringAxis)];

            if (endOffset < candidateEndOffset) {
                endOffset = candidateEndOffset;
            }

            if (startOffset > candidateStartOffset) {
                startOffset = candidateStartOffset;
            }

            crtClusterMemoryOffsets[Dim(clusteringAxis)] = startOffset;
            crtClusterMemoryShapes[Dim(clusteringAxis)] = endOffset - startOffset + 1;
        }
    }

    return OverlapDistributionParams(
            vpux::getIntArrayOfArray(ctx, memoryShapes), vpux::getIntArrayOfArray(ctx, memoryOffsets),
            vpux::getIntArrayOfArray(ctx, computeShapes), vpux::getIntArrayOfArray(ctx, computeOffsets));
}

OverlapDistributionParams vpux::VPU::getActivationOverlappedParams(VPU::ClusteredOpInterface clusteredOp,
                                                                   ArrayRef<int64_t> activationTensorNumTiles,
                                                                   vpux::NDTypeInterface inType) {
    const auto ctx = clusteredOp.getContext();

    const auto kernelTileAxis = extractKernelTileAxis(activationTensorNumTiles);
    const auto localOverlappedParams = getOverlappedDistributionParameters(
            ctx, SmallVector<VPU::ClusteredOpInterface>({clusteredOp}), kernelTileAxis);

    auto archKind = getArch(clusteredOp.getOperation());
    const std::set<VPU::ArchKind> compatibleTargets = {
            VPU::ArchKind::NPU40XX,
    };

    // For 30XX, 37XX, we do not set input workloads explicitly and therefore
    // OVERLAPPED should only represent the current op's input needs w/o
    // the sibling requirements
    if (compatibleTargets.count(archKind) != 1) {
        return localOverlappedParams;
    }

    SmallVector<VPU::ClusteredOpInterface> siblingSubgraph;

    // TODO: use common interference graph util
    const size_t numberOperands = mlir::isa<VPU::NCEEltwiseOp>(clusteredOp.getOperation()) ? 2 : 1;

    // propagate operations which have the same input and output overlapped params
    const auto isBypassOp = [&](mlir::Operation* op) {
        if (mlir::isa_and_nonnull<VPU::QuantizeCastOp>(op)) {
            return true;
        }

        if (auto eltwiseOp = mlir::dyn_cast_or_null<VPU::NCEEltwiseOp>(op)) {
            if (eltwiseOp.getIsInplace().value_or(false)) {
                return true;
            }
        }

        return false;
    };

    const auto findSibling = [&](mlir::Value operand) -> void {
        for (const auto& sibling : operand.getUsers()) {
            SmallVector<mlir::Operation*> userOps;
            if (isBypassOp(sibling)) {
                userOps = to_small_vector(sibling->getUsers());
            } else {
                userOps.push_back(sibling);
            }

            for (const auto& user : userOps) {
                if (auto clusteredUser = mlir::dyn_cast<VPU::ClusteredOpInterface>(user)) {
                    siblingSubgraph.push_back(clusteredUser);
                }
            }
        }
    };

    // Given the following subgraph:
    //             Producer
    //         /             |
    // [QuantizeCast]  [QuantizeCast]
    //       |               |
    //     NceOp0          NceOp1
    // NceOp0 and NceOp1 should be aware of each other as siblings to be able to properly set their input distributions
    for (size_t opIdx = 0; opIdx < numberOperands; ++opIdx) {
        auto operand = clusteredOp->getOperand(opIdx);
        findSibling(operand);

        // TODO: 104112 avoid spilling due to other view ops besides of QuantizeCast
        if (auto quantizeCastOp = mlir::dyn_cast_or_null<VPU::QuantizeCastOp>(operand.getDefiningOp())) {
            operand = quantizeCastOp->getOperand(0);
            findSibling(operand);
        }
    }

    const auto candidateOverlappedParams = getOverlappedDistributionParameters(ctx, siblingSubgraph, kernelTileAxis);

    // Check candidateOverlappedParams if valid
    // For example,
    // Conv 1 with Input_Y = 8
    // Conv 1 with [[kernel_Y 1, stride_Y 1, pad_top_bottom 0, 0]]
    // Conv 2 with [[kernel_Y 8, stride_Y 8, pad_top_bottom 0, 0]]
    //
    // candidateOverlappedParams will follow Conv 2
    // Conv 1 will have Output_Y = 1 when inferring output shape
    // => will fail to split over numTiles (usually >=2)
    // So we need have this check
    const auto numTilesPerDim = (kernelTileAxis == Dims4D::Kernel::Y.ind())
                                        ? activationTensorNumTiles[Dims4D::Act::H.ind()]
                                        : activationTensorNumTiles[Dims4D::Act::W.ind()];
    const auto inputShape = inType.getShape();
    auto isOverlappedParamsValidForSplit = [&]() {
        const std::pair<int64_t, int64_t> inputHW = {inputShape[Dims4D::Act::H], inputShape[Dims4D::Act::W]};
        VPUX_THROW_WHEN(candidateOverlappedParams.pads == nullptr, "Pads shouldn't be nullptr");
        const auto padInfo = toPadInfo(candidateOverlappedParams.pads);
        const auto getOutputHW = vpux::spatialOutputForInputWindowSize(inputHW, candidateOverlappedParams.kernel,
                                                                       candidateOverlappedParams.stride, padInfo);
        if (!getOutputHW.has_value()) {
            return false;
        }

        const auto outputHW = getOutputHW.value();
        return (kernelTileAxis == Dims4D::Kernel::Y.ind()) ? outputHW.first >= numTilesPerDim
                                                           : outputHW.second >= numTilesPerDim;
    };
    if (!isOverlappedParamsValidForSplit()) {
        return localOverlappedParams;
    }

    // Lacking a way specifying explicit per cluster shapes and offsets in the distributed
    // datatype, we are forced to pick a single sibling configuration.
    // Need to check if the picked configuration satisfies the current consumer
    // data requirements.
    //
    // For example two convolutions each with input height of 100 across 3 clusters
    // Conv 1 with [kernel_Y 1, stride_Y 1, pad_top_bottom 0, 0]
    // Conv 2 with [kernel_Y 5, stride_Y 4, pad_top_bottom 2, 2]
    //
    // Conv 1 needs [34, 33, 33] input size for each cluster
    // Conv 2 needs [35, 33, 31] input size for each cluster
    //
    // We can't pick a single sibling configuration to satisfy both operations.
    // Thus if the initial choice of config is not supporting, default to own config.
    //
    // This will change once we start using a more explicit representation for shapes per cluster
    // and no longer rely on the current distributed attribute config.

    const auto distributionModeAttr = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::OVERLAPPED);

    const auto candidateDistributedAttr = DistributedTensorAttr::get(
            ctx, distributionModeAttr, getIntArrayAttr(ctx, activationTensorNumTiles), candidateOverlappedParams.kernel,
            candidateOverlappedParams.pads, candidateOverlappedParams.stride, getIntAttr(ctx, numTilesPerDim), nullptr,
            mlir::UnitAttr::get(ctx), nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto localDistributedAttr = DistributedTensorAttr::get(
            ctx, distributionModeAttr, getIntArrayAttr(ctx, activationTensorNumTiles), localOverlappedParams.kernel,
            localOverlappedParams.pads, localOverlappedParams.stride, getIntAttr(ctx, numTilesPerDim), nullptr,
            mlir::UnitAttr::get(ctx), nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto candidateOffsets = getPerClusterMemoryShapeOffsets(inputShape, candidateDistributedAttr);
    const auto optionalCandidateShapes = getPerClusterMemoryShapes(inputShape, candidateDistributedAttr);
    VPUX_THROW_UNLESS(optionalCandidateShapes.has_value(),
                      "Cannot get per cluster memory shapes. Unsupported distribution: {0}", candidateDistributedAttr);
    const auto candidateShapes = optionalCandidateShapes.value();
    const auto localOffsets = getPerClusterMemoryShapeOffsets(inputShape, localDistributedAttr);
    const auto optionalLocalShapes = getPerClusterMemoryShapes(inputShape, localDistributedAttr);
    VPUX_THROW_UNLESS(optionalLocalShapes.has_value(),
                      "Cannot get per cluster memory shapes. Unsupported distribution: {0}", localDistributedAttr);
    const auto localShapes = optionalLocalShapes.value();

    for (auto offsetPerClusterZip : zip(candidateOffsets, localOffsets)) {
        for (auto dimZip : zip(std::get<0>(offsetPerClusterZip), std::get<1>(offsetPerClusterZip))) {
            if (std::get<0>(dimZip) > std::get<1>(dimZip)) {
                // candidate offset does not satisfy local op
                return localOverlappedParams;
            }
        }
    }

    const auto candidateEndOffset = getPerClusterEndOffset(candidateOffsets, candidateShapes);
    const auto localEndOffset = getPerClusterEndOffset(localOffsets, localShapes);

    for (auto endOffsetsPerClusterZip : zip(candidateEndOffset, localEndOffset)) {
        for (auto dimZip : zip(std::get<0>(endOffsetsPerClusterZip), std::get<1>(endOffsetsPerClusterZip))) {
            if (std::get<0>(dimZip) < std::get<1>(dimZip)) {
                // candidate shape does not satisfy local op
                return localOverlappedParams;
            }
        }
    }

    return candidateOverlappedParams;
}

std::set<VPU::ClusteredOpInterface> vpux::VPU::getSiblingOps(mlir::Operation* op) {
    if (!mlir::isa<VPU::ClusteredOpInterface>(op) && !isPassthroughOp(op)) {
        return {};
    }

    const bool isMultiInputOp = mlir::isa<VPU::NCEEltwiseOp>(op) || mlir::isa<VPU::ConcatOp>(op);

    std::set<VPU::ClusteredOpInterface> siblingSubgraph = {};
    std::set<llvm::hash_code> visitedTensors = {};
    for (auto operand : op->getOperands()) {
        findSiblings(operand, siblingSubgraph, visitedTensors);

        if (!isMultiInputOp) {
            break;
        }
    }

    return siblingSubgraph;
}

OverlapDistributionParams vpux::VPU::getActivationOverlappedParams(VPU::ClusteredOpInterface clusteredOp,
                                                                   ArrayRef<int64_t> activationTensorNumTiles,
                                                                   mlir::UnitAttr uniformDistributedSegments,
                                                                   vpux::NDTypeInterface inputType,
                                                                   const vpux::TileInfo& tileInfo) {
    const auto ctx = clusteredOp.getContext();
    auto archKind = getArch(clusteredOp.getOperation());
    const std::set<VPU::ArchKind> compatibleTargets = {
            VPU::ArchKind::NPU40XX,
    };

    // For 30XX, 37XX, we do not set input workloads explicitly and therefore
    // OVERLAPPED should only represent the current op's input needs w/o
    // the sibling requirements
    // E#106872 to remove arch check
    if (compatibleTargets.count(archKind) != 1) {
        const auto kernelTileAxis = extractKernelTileAxis(activationTensorNumTiles);
        return getOverlappedDistributionParameters(ctx, SmallVector<VPU::ClusteredOpInterface>({clusteredOp}),
                                                   kernelTileAxis);
    }

    // TODO: E#112803 Add support for extended memory view for sparse types with SETable
    auto sparseType = clusteredOp->getOperand(0).getType().dyn_cast<VPU::SparseTensorType>();
    const bool isSparseTypeInputWithSeTable = (sparseType != nullptr) && (sparseType.getSeAttr() != nullptr);
    const auto siblingSubgraph = (isSparseTypeInputWithSeTable) ? std::set<VPU::ClusteredOpInterface>{clusteredOp}
                                                                : getSiblingOps(clusteredOp.getOperation());

    const auto clusteringAxis = VPU::getDistributedTilingAxis(activationTensorNumTiles);
    return getOverlappedDistributionParameters(
            ctx, (inputType != nullptr) ? inputType : clusteredOp->getOperand(0).getType().cast<NDTypeInterface>(),
            SmallVector<VPU::ClusteredOpInterface>(siblingSubgraph.begin(), siblingSubgraph.end()),
            activationTensorNumTiles[clusteringAxis], activationTensorNumTiles, uniformDistributedSegments, tileInfo);
}

OverlapDistributionParams vpux::VPU::getOutputOverlappedParams(VPU::ClusteredOpInterface clusteredOp,
                                                               ArrayRef<int64_t> outputTensorNumTiles,
                                                               vpux::NDTypeInterface outputType,
                                                               ArrayRef<int64_t> activationTensorNumTiles) {
    const auto ctx = clusteredOp.getContext();
    SmallVector<VPU::ClusteredOpInterface> consumerSubgraph;
    auto archKind = getArch(clusteredOp.getOperation());
    const std::set<VPU::ArchKind> compatibleTargets = {
            VPU::ArchKind::NPU40XX,
    };
    const auto equalComputeAndMemoryView =
            compatibleTargets.count(archKind) <= 0 ? mlir::UnitAttr::get(clusteredOp.getContext()) : nullptr;

    if (auto eltwise = mlir::dyn_cast<VPU::NCEEltwiseOp>(clusteredOp.getOperation())) {
        if (eltwise.getIsInplace().value_or(false)) {
            // inplace eltwise should infer from input
            return getActivationOverlappedParams(clusteredOp, activationTensorNumTiles, outputType);
        }
    }

    for (const auto& result : clusteredOp->getResults()) {
        for (const auto& consumer : result.getUsers()) {
            if (auto clusteredConsumer = mlir::dyn_cast<VPU::ClusteredOpInterface>(consumer)) {
                consumerSubgraph.push_back(clusteredConsumer);
            }

            // Given the following subgraph:
            //     NCEProducer
            //         |
            //       Concat
            //      /    |
            // NceOp0  NceOp1
            // NCEProducer's effective consumers should be: {NceOp0, NceOp1}
            // NCEProducer should produce the largest halo possible to cover both NceOp0 & NceOp1 to allow the Concat
            // to be done implicitly in CMX.
            if (isValidCandidateForCMXConcat(consumer)) {
                for (const auto& consumerConcat : consumer->getUsers()) {
                    if (auto clusteredConsumerConcat = mlir::dyn_cast<VPU::ClusteredOpInterface>(consumerConcat)) {
                        consumerSubgraph.push_back(clusteredConsumerConcat);
                    }
                }
            }

            // Given the following subgraph:
            //     NCEProducer
            //         |
            //     QuantizeCast
            //         |
            //       NceOp
            // NCEProducer's effective consumer should be NceOp
            // TODO: 104112 avoid spilling due to other view ops besides of QuantizeCast
            if (auto quantizeCastConsumer = mlir::dyn_cast_or_null<VPU::QuantizeCastOp>(consumer)) {
                for (const auto& consumerQuantizeCast : quantizeCastConsumer->getUsers()) {
                    if (auto clusteredConsumerQuantizeCast =
                                mlir::dyn_cast<VPU::ClusteredOpInterface>(consumerQuantizeCast)) {
                        consumerSubgraph.push_back(clusteredConsumerQuantizeCast);
                    }
                }
            }

            // In-place Eltwise output overlapped params are inferred from input
            // propagate in-place Eltwise to consider consumers
            if (auto eltwiseConsumer = mlir::dyn_cast_or_null<VPU::NCEEltwiseOp>(consumer)) {
                if (eltwiseConsumer.getIsInplace().value_or(false)) {
                    for (const auto& consumerEltwise : eltwiseConsumer->getUsers()) {
                        if (auto clusteredConsumerEltwise =
                                    mlir::dyn_cast<VPU::ClusteredOpInterface>(consumerEltwise)) {
                            consumerSubgraph.push_back(clusteredConsumerEltwise);
                        }
                    }
                }
            }
        }
    }
    const auto kernelTileAxis = extractKernelTileAxis(outputTensorNumTiles);
    const auto candidateOverlappedParams = getOverlappedDistributionParameters(
            clusteredOp.getContext(), consumerSubgraph, kernelTileAxis, equalComputeAndMemoryView);

    // Lacking a way specifying explicit per cluster shapes and offsets in the distributed
    // datatype, we are forced to pick a configuration where compute view is within the boundaries
    // of the memory view.
    // We represent input workload start & end through compute offset and size, while the total
    // amount of data in cluster is represented through memory shape. In cases where
    // compute start < memory start or compute end > memory_end, the size of data in cluster should be
    // max(compute end, memory_end) - min(compute start, memory start) + 1, but we currently have no way of
    // representing that. Therefore, we ensure that such a case will not happen by setting overlapped params k1x1,
    // s1x1, pad0x0x0x0 if the consumer distribution does not satisfy the requirements.

    const auto numTilesPerDim = (kernelTileAxis == Dims4D::Kernel::Y.ind())
                                        ? outputTensorNumTiles[Dims4D::Act::H.ind()]
                                        : outputTensorNumTiles[Dims4D::Act::W.ind()];
    const auto distributionModeAttr = VPU::DistributionModeAttr::get(ctx, VPU::DistributionMode::OVERLAPPED);

    const auto candidateDistributedAttr = DistributedTensorAttr::get(
            ctx, distributionModeAttr, getIntArrayAttr(ctx, outputTensorNumTiles), candidateOverlappedParams.kernel,
            candidateOverlappedParams.pads, candidateOverlappedParams.stride, getIntAttr(ctx, numTilesPerDim), nullptr,
            mlir::UnitAttr::get(ctx), nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto kernel = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto pads = VPU::getPaddingAttr(ctx, 0, 0, 0, 0);
    const auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto fallbackOverlappedParams = OverlapDistributionParams(kernel, pads, strides, equalComputeAndMemoryView);

    const auto outputShape = (outputType == nullptr) ? getShape(clusteredOp->getResult(0)) : outputType.getShape();
    const auto optionalCandidateMemoryShapes = getPerClusterMemoryShapes(outputShape, candidateDistributedAttr);
    if (!optionalCandidateMemoryShapes.has_value()) {
        // If NCEProducer has tiling required and the tiled shape does not satisfy producer op
        return fallbackOverlappedParams;
    }

    const auto candidateMemoryOffsets = getPerClusterMemoryShapeOffsets(outputShape, candidateDistributedAttr);
    const auto candidateComputeOffsets = getPerClusterComputeShapeOffsets(outputShape, candidateDistributedAttr);
    const auto candidateComputeShapes = getPerClusterComputeShapes(outputShape, candidateDistributedAttr);

    // Memory start offset must be before or equal to compute start offset
    for (auto startOffsetsPerClusterZip : zip(candidateMemoryOffsets, candidateComputeOffsets)) {
        for (auto dimZip : zip(std::get<0>(startOffsetsPerClusterZip), std::get<1>(startOffsetsPerClusterZip))) {
            if (std::get<0>(dimZip) > std::get<1>(dimZip)) {
                // candidate shape does not satisfy producer op
                return fallbackOverlappedParams;
            }
        }
    }

    const auto candidateMemoryShapes = optionalCandidateMemoryShapes.value();
    const auto candidateMemoryEndOffset = getPerClusterEndOffset(candidateMemoryOffsets, candidateMemoryShapes);
    const auto candidateComputeEndOffset = getPerClusterEndOffset(candidateComputeOffsets, candidateComputeShapes);

    // Memory end offset must be after or equal to compute end offset
    for (auto endOffsetsPerClusterZip : zip(candidateMemoryEndOffset, candidateComputeEndOffset)) {
        for (auto dimZip : zip(std::get<0>(endOffsetsPerClusterZip), std::get<1>(endOffsetsPerClusterZip))) {
            if (std::get<0>(dimZip) < std::get<1>(dimZip)) {
                // candidate shape does not satisfy local op
                return fallbackOverlappedParams;
            }
        }
    }

    return candidateOverlappedParams;
}

OverlapDistributionParams vpux::VPU::getOutputOverlappedParams(VPU::ClusteredOpInterface clusteredOp,
                                                               ArrayRef<int64_t> outputTensorNumTiles,
                                                               mlir::UnitAttr uniformDistributedSegments,
                                                               vpux::NDTypeInterface outputType,
                                                               const vpux::TileInfo& tileInfo) {
    const auto ctx = clusteredOp.getContext();
    auto archKind = getArch(clusteredOp.getOperation());
    const std::set<VPU::ArchKind> compatibleTargets = {
            VPU::ArchKind::NPU40XX,
    };

    // For arch w/o halo support
    // E#106872 to remove arch check
    if (compatibleTargets.count(archKind) != 1) {
        VPUX_THROW_WHEN(!mlir::isa<NCEPermuteOp>(clusteredOp.getOperation()),
                        "Arch {0} does not support output OVERLAPPED distribution for op = {1}", archKind, clusteredOp);

        const auto equalComputeAndMemoryView = mlir::UnitAttr::get(clusteredOp.getContext());
        const auto kernelTileAxis = extractKernelTileAxis(outputTensorNumTiles);

        SmallVector<VPU::ClusteredOpInterface> sohOverlappedConsumer{};
        for (const auto consumer : clusteredOp->getUsers()) {
            if (auto clusteredConsumer = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(consumer)) {
                auto multiclusterStrategy = clusteredConsumer.getMultiClusterStrategy();
                if (!multiclusterStrategy.has_value()) {
                    continue;
                }

                if (multiclusterStrategy.value() == VPU::MultiClusterStrategy::SplitOverHeightOverlapped) {
                    sohOverlappedConsumer.push_back(clusteredConsumer);
                }
            }
        }

        return getOverlappedDistributionParameters(clusteredOp.getContext(), sohOverlappedConsumer, kernelTileAxis,
                                                   equalComputeAndMemoryView);
    }

    std::set<VPU::ClusteredOpInterface> consumers{};
    if (isPassthroughOp(clusteredOp.getOperation())) {
        // For passthrough ops, ensure input and output tensors use the same pool of ops to determine the distribution
        consumers.merge(getSiblingOps(clusteredOp.getOperation()));
    } else {
        for (const auto& consumer : clusteredOp->getUsers()) {
            // find first valid consumer and use it to get all its clustered siblings
            if (mlir::isa<VPU::ClusteredOpInterface>(consumer) || isPassthroughOp(consumer)) {
                consumers.merge(getSiblingOps(consumer));
                break;
            }
        }
    }

    const auto clusteringAxis = VPU::getDistributedTilingAxis(outputTensorNumTiles);
    return getOverlappedDistributionParameters(
            ctx, (outputType != nullptr) ? outputType : clusteredOp->getResult(0).getType().cast<NDTypeInterface>(),
            SmallVector<VPU::ClusteredOpInterface>(consumers.begin(), consumers.end()),
            outputTensorNumTiles[clusteringAxis], outputTensorNumTiles, uniformDistributedSegments, tileInfo);
}
