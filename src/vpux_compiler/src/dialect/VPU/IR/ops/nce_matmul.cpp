//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/Support/LogicalResult.h>

#include "vpux/compiler/dialect/IE/utils/matmul.hpp"
#include "vpux/compiler/dialect/VPU/IR/native_attributes/distributed_tensor_native.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/explicit_distribution_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/utils/analysis.hpp"

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/utils/core/logger.hpp"

using namespace vpux;

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEMatMulOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             std::optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties props,
                                                             [[maybe_unused]] mlir::RegionRange regions,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    NCEMatMulOpAdaptor op(operands, attrs, props);

    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }

    auto inputType = op.getInput().getType().cast<vpux::NDTypeInterface>();
    auto weightsType = op.getWeights().getType().cast<vpux::NDTypeInterface>();

    const auto inputShape = inputType.getShape();
    const auto weightsShape = weightsType.getShape();

    SmallVector<int64_t> outputShape{inputShape[Dim(0)], inputShape[Dim(1)], weightsShape[Dim(1)], inputShape[Dim(3)],
                                     inputShape[Dim(4)]};

    auto outputType =
            mlir::RankedTensorType::get(outputShape, inputType.getElementType(), createTensorAttrFromType(inputType));

    inferredReturnTypes.push_back(outputType);
    return mlir::success();
}

//
// Verifier
//

mlir::LogicalResult vpux::VPU::NCEMatMulOp::verify() {
    const auto op = getOperation();

    if (mlir::failed(vpux::VPU::verifyNCEOp(op))) {
        return mlir::failure();
    }

    return mlir::success();
}

//
// fitIntoCMX
//

bool doesNCEMatMulFitIntoCMX(vpux::NDTypeInterface inputType, vpux::NDTypeInterface filterType,
                             vpux::NDTypeInterface outputType, mlir::ModuleOp moduleOp, Byte reservedMem) {
    auto arch = VPU::getArch(moduleOp);

    auto largestGroupsNumPerCluster = filterType.getShape()[DimsGroups5D::Act::G];
    if (auto distType = mlir::dyn_cast<VPU::DistributedTensorType>(filterType)) {
        largestGroupsNumPerCluster = distType.getLargestCompactShape()[DimsGroups5D::Act::G];
    }

    const auto weightsTableSize = vpux::VPU::NCEInvariant::getWeightsTableSize(
            outputType.getShape()[DimsGroups5D::Act::C] * largestGroupsNumPerCluster);

    SmallVector<Byte> buffers = {
            inputType.getTotalAllocSize(),
            filterType.getTotalAllocSize(),
            outputType.getTotalAllocSize(),
            weightsTableSize,
    };

    const auto totalAvailableCMXSize = reservedMem.count() == 0
                                               ? vpux::VPU::getTotalCMXSize(moduleOp).count()
                                               : vpux::VPU::getTotalCMXFragmentationAwareSize(moduleOp).count();

    const auto requiredMemoryAligned = vpux::VPU::calculateAlignedBuffersMemoryRequirement(arch, buffers).count();

    return requiredMemoryAligned + reservedMem.count() <= totalAvailableCMXSize;
}

bool vpux::VPU::NCEMatMulOp::fitIntoCMX(vpux::NDTypeInterface inputType, vpux::NDTypeInterface filterType,
                                        vpux::NDTypeInterface outputType, Byte reservedMem) {
    auto mod = getModuleOp(getOperation());
    return doesNCEMatMulFitIntoCMX(inputType, filterType, outputType, mod, reservedMem);
}

bool vpux::VPU::NCEMatMulOp::fitIntoCMX(vpux::NDTypeInterface inputType, vpux::NDTypeInterface filterType,
                                        vpux::NDTypeInterface outputType) {
    return fitIntoCMX(inputType, filterType, outputType, Byte(0));
}

//
// isSupported
//

bool isNCEMatMulSupported(vpux::NDTypeInterface inputType, [[maybe_unused]] vpux::NDTypeInterface filterType,
                          vpux::NDTypeInterface outputType, mlir::ModuleOp moduleOp, vpux::LogCb logCb,
                          bool checkLayout, [[maybe_unused]] bool checkChannelAlignment) {
    if (auto inOrder = inputType.getDimsOrder(); checkLayout && inOrder != DimsOrder::GNHWC) {
        logCb(llvm::formatv("VPU::NCEMatMulOp input has unsupported layout '{0}'", inOrder));
        return false;
    }

    // If we have less groups than clusters, it doesn't make sense to try split-over-group optimisation.
    const auto groups = outputType.getShape()[DimsGroups5D::Act::G];
    const auto clusters = IE::getTileExecutor(moduleOp).getCount();

    if (groups < clusters) {
        logCb(llvm::formatv("VPU::NCEMatMulOp input has more groups than there are available clusters"));
        return false;
    }

    return true;
}

bool VPU::NCEMatMulOp::isSupported(IE::MatMulOp op, vpux::LogCb logCb, bool checkLayout, bool checkChannelAlignment) {
    auto mod = getModuleOp(op);

    const auto inputType = op.getInput1().getType().cast<vpux::NDTypeInterface>();
    const auto filterType = op.getInput2().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = op.getOutput().getType().cast<vpux::NDTypeInterface>();

    const auto inputShape = inputType.getShape();
    const auto filterShape = filterType.getShape();
    const auto outputShape = outputType.getShape();

    bool isSupported = isNCEMatMulSupported(
            inputType.changeShape(Shape({inputShape[Dims4D::Act::C] * inputShape[Dims4D::Act::N], 1,
                                         inputShape[Dims4D::Act::W], inputShape[Dims4D::Act::H], 1})),
            // Filter shape 2nd and 3rd can be incorrect depending on transposeB option, however filter is not used in
            // checks
            filterType.changeShape(Shape({filterShape[Dims4D::Act::C] * filterShape[Dims4D::Act::N],
                                          filterShape[Dims4D::Act::H], filterShape[Dims4D::Act::W], 1, 1})),
            outputType.changeShape(Shape({outputShape[Dims4D::Act::C] * outputShape[Dims4D::Act::N], 1,
                                          outputShape[Dims4D::Act::W], outputShape[Dims4D::Act::H], 1})),
            mod, logCb, checkLayout, checkChannelAlignment);
    isSupported = isSupported && IE::doesIEMatMulFitIntoCMX(op, inputShape, filterShape);
    return isSupported;
}

bool VPU::NCEMatMulOp::isSupported(VPU::NCEMatMulOp op, vpux::LogCb logCb, bool checkLayout,
                                   bool checkChannelAlignment) {
    auto mod = getModuleOp(op);

    const auto inputType = op.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto filterType = op.getWeights().getType().cast<vpux::NDTypeInterface>();
    const auto outputType = op.getOutput().getType().cast<vpux::NDTypeInterface>();

    return isNCEMatMulSupported(inputType, filterType, outputType, mod, logCb, checkLayout, checkChannelAlignment);
}

//
// TilingBuilderOpInterace
//

TilingInfo vpux::VPU::NCEMatMulOp::backInferTileInfo([[maybe_unused]] const vpux::TileInfo& outputTile,
                                                     [[maybe_unused]] vpux::Logger log) {
    VPUX_THROW("VPU::NCEMatMulOp::backInferTileInfo is not implemented!");
}

void vpux::VPU::NCEMatMulOp::adjustAttrs([[maybe_unused]] const vpux::TilingInfo&,
                                         [[maybe_unused]] const vpux::TileInfo& outputTile) {
    VPUX_THROW("VPU::NCEMatMulOp::adjustAttrs is not implemented!");
}

mlir::FailureOr<OutputTiling> vpux::VPU::NCEMatMulOp::getTilingStrategy([[maybe_unused]] TilingMode tilingMode,
                                                                        [[maybe_unused]] Logger log) {
    VPUX_THROW("VPU::NCEMatMulOp::getTilingStrategy is not implemented!");
}

//
// ClusteredOpInterface
//

bool vpux::VPU::NCEMatMulOp::checkStrategyCompatibility(vpux::VPU::MultiClusterStrategy strategy, size_t) {
    return strategy == VPU::MultiClusterStrategy::SplitOverGroup;
}

vpux::VPU::DistributedTensorNative vpux::VPU::NCEMatMulOp::getExplicitDistributedTensorAttr(
        vpux::ShapeRef shape, vpux::VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
        const int64_t numClusters, ArrayRef<int64_t> alignment, const bool uniformDistributedSegments,
        const vpux::VPU::OverlapDistributionParams& overlapParams) {
    return VPU::getNCEExplicitDistributedTensorNative(mlir::dyn_cast<VPU::NCEOpInterface>(getOperation()), shape,
                                                      distributionMode, numTiles, numClusters, alignment,
                                                      uniformDistributedSegments, overlapParams);
}

bool VPU::NCEMatMulOp::isOperationSplitOverHeightCompatible([[maybe_unused]] const vpux::TileInfo& oriOutputTile) {
    return false;
}

bool VPU::NCEMatMulOp::isOperationSplitOverWidthCompatible([[maybe_unused]] ShapeRef outputShape,
                                                           [[maybe_unused]] ShapeRef offset,
                                                           [[maybe_unused]] ShapeRef axis) {
    return false;
}

bool VPU::NCEMatMulOp::isOperationSplitOverKernelCompatible([[maybe_unused]] ShapeRef outputShape,
                                                            [[maybe_unused]] ShapeRef offset,
                                                            [[maybe_unused]] ShapeRef axis) {
    return false;
}

bool VPU::NCEMatMulOp::doesLayerFitIntoCMX(VPU::MultiClusterStrategy strategy, Byte reservedMem) {
    auto nceOp = mlir::cast<VPU::NCEMatMulOp>(getOperation());
    auto nceOpInterface = mlir::cast<VPU::NCEOpInterface>(getOperation());
    const auto outputType = nceOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    auto numClusters = VPU::getOptimalNumClusters(nceOp, outputType.getShape(), strategy);
    auto mod = getModuleOp(getOperation());
    auto arch = VPU::getArch(mod);

    auto filterType = getWeights().getType().cast<vpux::NDTypeInterface>();
    auto largestGroupsNumPerCluster = filterType.getShape()[DimsGroups5D::Act::G];
    if (auto distType = mlir::dyn_cast<VPU::DistributedTensorType>(filterType)) {
        largestGroupsNumPerCluster = distType.getLargestCompactShape()[DimsGroups5D::Act::G];
    }

    const auto weightsTableSize = vpux::VPU::NCEInvariant::getWeightsTableSize(
            outputType.getShape()[DimsGroups5D::Act::C] * largestGroupsNumPerCluster);

    SmallVector<Byte> buffers = {
            VPU::getTotalAllocSizeWithDistribution(
                    getInput().getType(),
                    getActivationDistributionAttrFromOp(nceOp, getInput().getType(), numClusters.getInt(), strategy)),
            VPU::getTotalAllocSizeWithDistribution(
                    getWeights().getType(), getFilterDistributionAttrFromOp(nceOpInterface, getWeights().getType(),
                                                                            numClusters.getInt(), strategy)),
            VPU::getTotalAllocSizeWithDistribution(
                    getOutput().getType(),
                    getOutputDistributionAttrFromOp(nceOp, getOutput().getType(), numClusters.getInt(), strategy)),
            weightsTableSize};

    const auto totalAvailableCMXSize = reservedMem.count() == 0
                                               ? vpux::VPU::getTotalCMXSize(mod).count()
                                               : vpux::VPU::getTotalCMXFragmentationAwareSize(mod).count();

    const auto requiredMemoryAligned = vpux::VPU::calculateAlignedBuffersMemoryRequirement(arch, buffers).count();

    return requiredMemoryAligned + reservedMem.count() <= totalAvailableCMXSize;
}

bool VPU::NCEMatMulOp::doesLayerChangeOutputAlignmentFitIntoCMX(
        VPU::MultiClusterStrategy strategy, VPU::DistributedTypeInterface newDistributedTensorType) {
    auto nceOp = mlir::cast<VPU::NCEMatMulOp>(getOperation());
    auto nceOpInterface = mlir::cast<VPU::NCEOpInterface>(getOperation());
    auto numClusters = VPU::getOptimalNumClusters(
            nceOp, nceOp.getOutput().getType().cast<vpux::NDTypeInterface>().getShape(), strategy);
    auto distributedInputType =
            getDistributedActivationTypeFromOp(nceOp, nceOp.getInput().getType(), numClusters, strategy);
    auto distributedFilterType =
            getDistributedFilterTypeFromOp(nceOpInterface, nceOp.getWeights().getType(), numClusters, strategy);
    return fitIntoCMX(distributedInputType, distributedFilterType, newDistributedTensorType);
}
