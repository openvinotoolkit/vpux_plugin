//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <set>
#include <unordered_map>
#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/native_attributes/distributed_tensor_native.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux {
namespace VPU {

constexpr int64_t KMB_DPU_CHANNELS_ALIGNMENT = 16;
constexpr StringLiteral multiClusterStrategy = "multiClusterStrategy";
const SmallVector<int64_t> DISTRIBUTED_C_ALIGNMENT = SmallVector<int64_t>{1, 16, 1, 1};

VPU::DistributedTensorAttr updateSliceLikeOpsAlignment(mlir::MLIRContext* ctx, vpux::ShapeRef inShape,
                                                       vpux::ShapeRef sliceShape,
                                                       VPU::DistributedTensorAttr originDistribution);
bool isSOCSegmentedOp(mlir::Operation* op);
bool isSOCSegmentedSWOp(mlir::Operation* op);
bool isSOCSegmentedNCEOp(mlir::Operation* op);
bool inputProducersCompatible(mlir::Operation* op, mlir::DenseSet<mlir::Operation*> handledUsers = {});
bool isSegmentedInputCompatible(mlir::Operation* op, mlir::DenseSet<mlir::Operation*> handledUsers = {});
bool isSOKSegmentedOutputCompatible(mlir::Operation* op);
int64_t getNumberOfClustersForSOKToAvoidAlignment(int64_t outputChannels, int64_t numClustersForCompilation,
                                                  bool uniformDistributedSegments = true);
int64_t getNumberOfClustersForSpatialDim(int64_t outputSpatialDim, int64_t numClustersForCompilation,
                                         bool uniformDistributedSegments = true);
SmallVector<int64_t> getActivationTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                                 int64_t numClustersAvailableForCompilation,
                                                 VPU::MultiClusterStrategy strategy);
std::optional<SmallVector<int64_t>> getActivationTensorAlignment(VPU::ClusteredOpInterface clusteredOp,
                                                                 int64_t numClusters,
                                                                 VPU::MultiClusterStrategy strategy,
                                                                 vpux::NDTypeInterface inputType = nullptr,
                                                                 vpux::NDTypeInterface outputType = nullptr);
SmallVector<int64_t> getOutputTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                             int64_t numClustersAvailableForCompilation,
                                             VPU::MultiClusterStrategy strategy);
std::optional<SmallVector<int64_t>> getOutputTensorAlignment(VPU::MultiClusterStrategy strategy);
std::optional<vpux::NDTypeInterface> adjustOutputAlignmentForSOH(VPU::ClusteredOpInterface clusteredOp,
                                                                 vpux::NDTypeInterface originalDistType);

SmallVector<int64_t> getWeightsTensorNumTiles(VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface tensorType,
                                              int64_t numClustersAvailableForCompilation,
                                              VPU::MultiClusterStrategy strategy);
std::optional<SmallVector<int64_t>> getWeightsTensorAlignment(VPU::MultiClusterStrategy strategy);
SmallVector<int64_t> getWeightsTableTensorNumTiles(VPU::ClusteredOpInterface clusteredOp,
                                                   vpux::NDTypeInterface tensorType,
                                                   int64_t numClustersAvailableForCompilation,
                                                   VPU::MultiClusterStrategy strategy);
SmallVector<int64_t> getInstructionListTableTensorNumTiles(VPU::MultiClusterStrategy strategy);
SmallVector<int64_t> getActivationWindowTensorNumTiles(VPU::MultiClusterStrategy strategy);
DistributionMode getActivationTensorDistributionMode(VPU::ClusteredOpInterface clusteredOp,
                                                     VPU::MultiClusterStrategy strategy);
DistributionMode getWeightsTensorDistributionMode(VPU::MultiClusterStrategy strategy);
DistributionMode getInstructionListTableTensorDistributionMode(VPU::MultiClusterStrategy strategy);
DistributionMode getOutputTensorDistributionMode(VPU::ClusteredOpInterface clusteredOp,
                                                 VPU::MultiClusterStrategy strategy);
DistributionMode getActivationWindowTensorDistributionMode(VPU::MultiClusterStrategy strategy);

int64_t getSOHPerClusterHeightAlignment(int64_t inputWidth, bool isInputSparse);
int64_t getSOHMinimalHeightAlignment(vpux::ShapeRef shape, int64_t numClusters, bool isInputSparse, VPU::ArchKind arch);
bool isSOHSupportedByDPU(vpux::NDTypeInterface inputType, ShapeRef inputShape, int64_t numClusters, bool DWTypeOp,
                         VPU::ArchKind arch);
bool isSOGSupportedByDPU(vpux::NDTypeInterface inputType, ShapeRef inputShape, int64_t numClusters, bool DWTypeOp,
                         VPU::ArchKind arch);

vpux::VPU::CopyOp createDistributedCopyIn(mlir::PatternRewriter& rewriter, VPU::ClusteredOpInterface clusteredOp,
                                          mlir::Value input, DistributionMode distributionMode,
                                          mlir::ArrayAttr numTiles, mlir::ArrayAttr alignment,
                                          VPU::MultiClusterStrategy strategy, bool hasExplicitDistributedAttr);

vpux::NDTypeInterface getDistributedTypeFromInput(VPU::ClusteredOpInterface clusteredOp, mlir::Value input,
                                                  DistributionMode distributionMode, mlir::ArrayAttr numTiles,
                                                  mlir::ArrayAttr alignment, VPU::MultiClusterStrategy strategy,
                                                  bool hasExplicitDistributedAttr);

bool getUniformDistributedSegments(VPU::ClusteredOpInterface clusteredOp, ArrayRef<int64_t> shape,
                                   VPU::DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
                                   ArrayRef<int64_t> alignment);

VPU::DistributedTensorType createExplicitDistributedTensorType(
        VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface inputType, DistributionMode distributionMode,
        ArrayRef<int64_t> numTiles, mlir::IntegerAttr numClusters, ArrayRef<int64_t> alignment,
        const bool uniformDistributedSegments, const VPU::OverlapDistributionParams& overlapParams);

VPU::DistributedTensorType createDistributedTensorType(VPU::ClusteredOpInterface clusteredOp,
                                                       vpux::NDTypeInterface inputType,
                                                       DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
                                                       mlir::IntegerAttr numClusters, ArrayRef<int64_t> alignment,
                                                       bool uniformDistributedSegments, bool hasExplicitDistributedAttr,
                                                       const VPU::OverlapDistributionParams& overlapParams);

VPU::DistributedTensorType createDistributedTensorType(VPU::NCEOpInterface nceOp, vpux::NDTypeInterface inputType,
                                                       DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
                                                       mlir::IntegerAttr numClusters, ArrayRef<int64_t> alignment,
                                                       bool uniformDistributedSegments, ArrayRef<int64_t> kernel = {},
                                                       VPU::PaddingAttr pad = nullptr, ArrayRef<int64_t> stride = {},
                                                       bool equalComputeAndMemoryView = false);

VPU::SparseTensorType createSparseTensorDistributedType(VPU::ClusteredOpInterface clusteredOp,
                                                        VPU::SparseTensorType sparseInputType,
                                                        DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
                                                        mlir::IntegerAttr numClusters, ArrayRef<int64_t> alignment,
                                                        bool uniformDistributedSegments,
                                                        bool hasExplicitDistributedAttr,
                                                        const VPU::OverlapDistributionParams& overlapParams);

VPU::DistributedTensorType createDistributedTensorType(mlir::Operation* viewLikeOp, vpux::NDTypeInterface inputType,
                                                       DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
                                                       mlir::IntegerAttr optimalNumberOfClusters,
                                                       ArrayRef<int64_t> alignment, bool uniformDistributedSegments,
                                                       ArrayRef<int64_t> kernel = {}, VPU::PaddingAttr pad = nullptr,
                                                       ArrayRef<int64_t> stride = {});

VPU::DistributedTensorType createDistributedTensorType(VPU::SWOpInterface swOp, vpux::NDTypeInterface inputType,
                                                       DistributionMode distributionMode, ArrayRef<int64_t> numTiles,
                                                       mlir::IntegerAttr numClusters, ArrayRef<int64_t> alignment,
                                                       bool uniformDistributedSegments);

VPU::DistributedTypeInterface getDistributedActivationTypeFromOp(
        VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface inputType, mlir::IntegerAttr numClusters,
        vpux::NDTypeInterface tiledOutputType = nullptr, const vpux::TileInfo& tileInfo = vpux::TileInfo(ShapeRef()));

VPU::DistributedTypeInterface getDistributedFilterTypeFromOp(VPU::NCEOpInterface nceOp, vpux::NDTypeInterface inputType,
                                                             mlir::IntegerAttr numClusters);

VPU::DistributedTypeInterface getDistributedOutputTypeFromOp(
        VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface outputType, mlir::IntegerAttr numClusters,
        vpux::NDTypeInterface inputType = nullptr, const vpux::TileInfo& tileInfo = vpux::TileInfo(ShapeRef()),
        bool hasExplicitDistributedAttr = false,
        const std::optional<OverlapDistributionParams>& overlappedParams = std::nullopt);

VPU::DistributedTypeInterface getDistributedActivationTypeFromOp(
        VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface inputType, mlir::IntegerAttr numClusters,
        VPU::MultiClusterStrategy customStrategy, ArrayRef<int64_t> customAlignment = {},
        vpux::NDTypeInterface tiledOutputType = nullptr, const vpux::TileInfo& tileInfo = vpux::TileInfo(ShapeRef()));

VPU::DistributedTypeInterface getDistributedFilterTypeFromOp(VPU::NCEOpInterface nceOp, vpux::NDTypeInterface inputType,
                                                             mlir::IntegerAttr numClusters,
                                                             VPU::MultiClusterStrategy customStrategy);

VPU::DistributedTypeInterface getDistributedOutputTypeFromOp(
        VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface outputType, mlir::IntegerAttr numClusters,
        VPU::MultiClusterStrategy customStrategy, vpux::NDTypeInterface inputType = nullptr,
        const vpux::TileInfo& tileInfo = vpux::TileInfo(ShapeRef()), bool hasExplicitDistributedAttr = false,
        const std::optional<OverlapDistributionParams>& overlappedParams = std::nullopt);

vpux::NDTypeInterface getDistributedOutputTensorType(
        VPU::ClusteredOpInterface clusteredOp, mlir::IntegerAttr numClusters, VPU::MultiClusterStrategy strategy,
        vpux::NDTypeInterface outputTensorType, bool hasExplicitDistributedAttr, bool alignForSOH = true,
        const std::optional<OverlapDistributionParams>& overlappedParams = std::nullopt);

OverlapDistributionParams getOverlapDistributionParams(VPU::ClusteredOpInterface clusteredOp,
                                                       vpux::NDTypeInterface outputTensorType,
                                                       std::set<VPU::ClusteredOpInterface>& opSiblings,
                                                       VPU::MultiClusterStrategy strategy);

bool isSegmentedOverlappedAxisSameAsSliceAxis(mlir::ArrayAttr numTiles, ArrayRef<int64_t> inputShape,
                                              ArrayRef<int64_t> sliceShape);

bool isSegmentedOverlappedAxisSameAsSliceAxis(ArrayRef<int64_t> numTiles, ArrayRef<int64_t> inputShape,
                                              ArrayRef<int64_t> sliceShape);

mlir::Type getCompactTypeFromDistributed(mlir::Type originalType);

Shape getLargestClusterOutputShape(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy);
bool isDWOpAndNeedsAlign(ArchKind arch, VPUIP::NCETaskType nceTaskType);
bool isEltwiseOpAndNeedsAlign(VPU::ClusteredOpInterface nceOp);
bool isSWOpChannelAlignmentCompatible(VPU::ClusteredOpInterface swOp, vpux::NDTypeInterface inputType,
                                      vpux::NDTypeInterface outputType);
bool isSWOpWithAlignedInputChannelReq(VPU::ClusteredOpInterface swOp, vpux::NDTypeInterface inputType = nullptr,
                                      vpux::NDTypeInterface outputType = nullptr);
bool isSWOpWithAlignedOutputChannelReq(VPU::ClusteredOpInterface swOp, vpux::NDTypeInterface inputType = nullptr,
                                       vpux::NDTypeInterface outputType = nullptr);

VPU::DistributedTensorType composeDistributedType(VPU::ClusteredOpInterface permuteOp,
                                                  VPU::DistributedTensorType distType, vpux::NDTypeInterface ndType,
                                                  mlir::ArrayAttr tileOverDim,
                                                  const OverlapDistributionParams& fusedOverlapParams,
                                                  bool enableExplicitDistributedTensorAttr = false,
                                                  bool equalComputeAndMemoryView = false);

mlir::Operation* getNextCompressConv(mlir::Operation* nceOp);
mlir::Type fuseOverlapParams(VPU::ClusteredOpInterface permuteOp, VPU::DistributedTensorType distType,
                             mlir::Operation* nextConv, bool enableExplicitDistributedTensorAttr = false);

template <typename T, std::enable_if_t<std::is_same<VPU::NCEClusterTilingOp, T>::value, bool> = true>
mlir::Value getDistributedOperandFromNCEClusterTiling(T clusterOp, mlir::Value innerOperand) {
    if (innerOperand == nullptr) {
        return nullptr;
    }
    auto blockArg = innerOperand.dyn_cast<mlir::BlockArgument>();
    if (blockArg == nullptr) {
        return nullptr;
    }
    auto operandNum = blockArg.getArgNumber();
    VPUX_THROW_UNLESS(operandNum < clusterOp.getNumOperands(),
                      "Argument number '{0}' is out of range of operands for NCEClusterTiling op '{1}'", operandNum,
                      clusterOp.getNumOperands());
    return clusterOp.getOperand(operandNum);
}

/**
 * @brief Check if the distributed mode is SEGMENTED
 *        or OVERLAPPED but no actual overlapped region
 */
template <typename T, std::enable_if_t<or_<std::is_same<VPU::DistributedTensorType, T>,
                                           std::is_same<VPUIP::DistributedBufferType, T>>::value,
                                       bool> = true>
bool isSegmentedLikeMode(T distributedType) {
    if (distributedType == nullptr || distributedType.getDistribution() == nullptr) {
        return false;
    }
    const auto distributionAttr = distributedType.getDistribution();
    const auto distributionMode = distributionAttr.getMode().getValue();
    if (distributionMode == DistributionMode::SEGMENTED) {
        return true;
    }
    if (distributionMode != DistributionMode::OVERLAPPED) {
        return false;
    }
    const auto ctx = distributedType.getContext();
    // Check if OVERLAPPED per cluster shapes and offsets are identical to the SEGMENTED mode
    const auto segmentedDistributionModeAttr = DistributionModeAttr::get(ctx, VPU::DistributionMode::SEGMENTED);
    const auto numTilesAttr = distributionAttr.getNumTiles();
    const auto ndType = distributedType.template cast<vpux::NDTypeInterface>();
    if (numTilesAttr == nullptr) {
        return false;
    }

    const auto segmentedDistributedAttr = getNonOverlappedDistributedAttr(
            ndType.getShape(), segmentedDistributionModeAttr, numTilesAttr, distributionAttr.getNumClusters(),
            distributionAttr.getAlignment(), distributionAttr.getUniformDistributedSegments(), ctx);

    return arePerClusterMemoryShapeAndOffsetsEqual(distributedType, segmentedDistributedAttr);
}

/**
 * @brief OVERLAPPED cluster tiling is only supported for dimensions H and W
 *        If it is actually SEGMENTED, this function can be used to replace the mode with SEGMENTED
 */
mlir::FailureOr<VPU::DistributedTensorAttr> legalizeCastedDistribution(VPU::DistributedTensorAttr castedDistribution,
                                                                       mlir::MLIRContext* ctx);

//
// Create DistributedTensorAttr
//

VPU::DistributedTensorNative createDistributedTensorNative(
        VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface inputType, DistributionMode distributionMode,
        ArrayRef<int64_t> numTiles, int64_t numClusters, ArrayRef<int64_t> alignment, bool uniformDistributedSegments,
        bool hasExplicitDistributedAttr, const VPU::OverlapDistributionParams& overlapParams);

VPU::DistributedTensorNative createDistributedTensorNative(VPU::NCEOpInterface nceOp, DistributionMode distributionMode,
                                                           ArrayRef<int64_t> numTiles, int64_t numClusters,
                                                           ArrayRef<int64_t> alignment, bool uniformDistributedSegments,
                                                           ArrayRef<int64_t> kernel = {},
                                                           const std::optional<VPU::Padding>& pad = std::nullopt,
                                                           ArrayRef<int64_t> stride = {},
                                                           bool equalComputeAndMemoryView = false);

VPU::DistributedTensorNative createDistributedTensorNative(mlir::Operation* viewLikeOp,
                                                           DistributionMode distributionMode,
                                                           ArrayRef<int64_t> numTiles, int64_t optimalNumberOfClusters,
                                                           ArrayRef<int64_t> alignment, bool uniformDistributedSegments,
                                                           ArrayRef<int64_t> kernel = {},
                                                           const std::optional<VPU::Padding>& pad = std::nullopt,
                                                           ArrayRef<int64_t> stride = {});

VPU::DistributedTensorNative createDistributedTensorNative(VPU::SWOpInterface swOp, DistributionMode distributionMode,
                                                           ArrayRef<int64_t> numTiles, int64_t optimalNumberOfClusters,
                                                           ArrayRef<int64_t> alignment,
                                                           bool uniformDistributedSegments);

VPU::DistributedTensorNative composeDistributedAttr(VPU::ClusteredOpInterface permuteOp,
                                                    VPU::DistributedTensorType distType, vpux::NDTypeInterface ndType,
                                                    mlir::ArrayAttr tileOverDim,
                                                    const OverlapDistributionParams& fusedOverlapParams,
                                                    bool enableExplicitDistributedTensorAttr = false,
                                                    bool equalComputeAndMemoryView = false);

llvm::DenseMap<mlir::Type, VPU::DistributedTensorNative> getOutputDistributionAttrFromOp(
        VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface outputType, int64_t numClusters,
        VPU::MultiClusterStrategy customStrategy, vpux::NDTypeInterface inputType = nullptr,
        const vpux::TileInfo& tileInfo = vpux::TileInfo(ShapeRef()), bool hasExplicitDistributedAttr = false);

llvm::DenseMap<mlir::Type, VPU::DistributedTensorNative> getActivationDistributionAttrFromOp(
        VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface inputType, int64_t numClusters,
        vpux::NDTypeInterface tiledOutputType = nullptr, const vpux::TileInfo& tileInfo = vpux::TileInfo(ShapeRef()));

llvm::DenseMap<mlir::Type, VPU::DistributedTensorNative> getOutputDistributionAttrFromOp(
        VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface outputType, int64_t numClusters,
        vpux::NDTypeInterface inputType = nullptr, const vpux::TileInfo& tileInfo = vpux::TileInfo(ShapeRef()),
        bool hasExplicitDistributedAttr = false);

llvm::DenseMap<mlir::Type, VPU::DistributedTensorNative> getActivationDistributionAttrFromOp(
        VPU::ClusteredOpInterface clusteredOp, vpux::NDTypeInterface inputType, int64_t numClusters,
        VPU::MultiClusterStrategy customStrategy, ArrayRef<int64_t> customAlignment = {},
        vpux::NDTypeInterface tiledOutputType = nullptr, const vpux::TileInfo& tileInfo = vpux::TileInfo(ShapeRef()));

llvm::DenseMap<mlir::Type, VPU::DistributedTensorNative> getFilterDistributionAttrFromOp(
        VPU::NCEOpInterface nceOp, vpux::NDTypeInterface inputType, int64_t numClusters,
        VPU::MultiClusterStrategy customStrategy);

vpux::Byte getTotalAllocSizeWithDistribution(vpux::NDTypeInterface type,
                                             const VPU::DistributedTensorNative& distribution);

vpux::Byte getTotalAllocSizeWithDistribution(
        vpux::NDTypeInterface type, const llvm::DenseMap<mlir::Type, VPU::DistributedTensorNative>& distributions);

}  // namespace VPU
}  // namespace vpux
