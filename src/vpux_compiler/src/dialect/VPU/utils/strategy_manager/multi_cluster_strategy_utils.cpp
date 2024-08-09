//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_infer.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/sparsity_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/strategy_manager/strategy_manager.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/TypeSwitch.h>

#include <unordered_map>

using namespace vpux;
using namespace VPU;

namespace {

double getSpillingCostForNonMultiCluster(vpux::NDTypeInterface tensorType, SpillingType /*spillingType*/,
                                         double ddrLatency, double ddrBandwidth, double /*cmxLatency*/,
                                         double /*cmxBandwidth*/, int64_t /*numDMAPorts*/) {
    // calculate the data byte size need copy from cmx to ddr or vice versa
    const auto totalSize = static_cast<double>(tensorType.getTotalAllocSize().count());
    return ddrLatency + totalSize / ddrBandwidth;
}

double getSpillingCostForDuplicated(vpux::NDTypeInterface tensorType, SpillingType /*spillingType*/, double ddrLatency,
                                    double ddrBandwidth, double /*cmxLatency*/, double /*cmxBandwidth*/,
                                    int64_t /*numDMAPorts*/) {
    auto distributedTensorType = tensorType.dyn_cast<DistributedTensorType>();
    VPUX_THROW_WHEN(distributedTensorType == nullptr, "Invalid type: {0}", tensorType);
    const auto totalSize = tensorType.getTotalAllocSize().count();
    return ddrLatency + totalSize / ddrBandwidth;
}

double getSpillingCostForSegmented(vpux::NDTypeInterface tensorType, SpillingType, double ddrLatency,
                                   double ddrBandwidth, double, double, int64_t numDMAPorts) {
    auto distributedTensorType = tensorType.dyn_cast<DistributedTensorType>();
    VPUX_THROW_WHEN(distributedTensorType == nullptr, "Invalid type: {0}", tensorType);
    const auto perClusterMemShapes = distributedTensorType.getPerClusterMemoryShapes();

    // Aggregate the total size which needs to be transfered on each DMA port
    auto totalSizeOnPorts = SmallVector<int64_t>(numDMAPorts, 0);
    for (size_t i = 0; i < perClusterMemShapes.size(); ++i) {
        totalSizeOnPorts[i % numDMAPorts] += perClusterMemShapes[i].totalSize();
    }
    // Considering multiple ports used in parallel, only take into account the largest size to transfer
    auto totalSize = *std::max_element(totalSizeOnPorts.begin(), totalSizeOnPorts.end());

    const Bit elemSize = tensorType.getElemTypeSize();
    totalSize = alignMemSize(elemSize * totalSize, Byte(1)).to<Byte>().count();
    return ddrLatency + static_cast<double>(totalSize) / ddrBandwidth;
}

using GetSpillingCostCB = double (*)(vpux::NDTypeInterface, SpillingType, double ddrLatency, double ddrBandwidth,
                                     double cmxLatency, double cmxBandwidth, int64_t numDMAPorts);
const EnumMap<DistributionMode, GetSpillingCostCB> spillingCostMap{
        // using  DistributionMode::NONE for single clustering case
        {DistributionMode::NONE, getSpillingCostForNonMultiCluster},
        {DistributionMode::DUPLICATED, getSpillingCostForDuplicated},
        {DistributionMode::SEGMENTED, getSpillingCostForSegmented},
        {DistributionMode::OVERLAPPED, getSpillingCostForSegmented},
        {DistributionMode::MULTICASTED, getSpillingCostForDuplicated},
        {DistributionMode::DUPLICATED | DistributionMode::SEGMENTED, getSpillingCostForDuplicated},
        {DistributionMode::MULTICASTED | DistributionMode::SEGMENTED, getSpillingCostForDuplicated},
};

mlir::Value getInputFromClusteredOp(VPU::ClusteredOpInterface clusteredOp, mlir::Operation* parentOp) {
    for (auto operand : clusteredOp->getOperands()) {
        auto parent = operand.getDefiningOp();
        if (parent == parentOp) {
            return operand;
        }
        while (mlir::isa_and_nonnull<VPU::DistributedCastOpInterface, VPU::ShapeCastOp, VPU::GroupSparseTensorOp>(
                parent)) {
            // propagate cast ops
            parent = parent->getOperand(0).getDefiningOp();
            if (parent == parentOp) {
                return operand;
            }
        }
    }

    VPUX_THROW("Cannot find input from op: {0}, parent op: {1}", clusteredOp, parentOp);
}

bool hasUserMVN(VPU::ClusteredOpInterface clusteredOp) {
    if (!clusteredOp->getOperand(0).isa<mlir::BlockArgument>() &&
        mlir::isa<VPU::MVNOp>(clusteredOp->getOperand(0).getDefiningOp())) {
        // MVN producer
        return true;
    }
    for (auto* user : clusteredOp->getResult(0).getUsers()) {
        // MVN consumer
        if (mlir::isa<VPU::MVNOp>(user)) {
            return true;
        }
    }
    return false;
}

bool isSOHAlignmentCompatibleOrAdjustedCompatible(vpux::NDTypeInterface srcType, vpux::NDTypeInterface dstType) {
    const auto srcTypeInterface = srcType.dyn_cast<VPU::DistributedTypeInterface>();
    const auto dstTypeInterface = dstType.dyn_cast<VPU::DistributedTypeInterface>();
    if (srcTypeInterface == nullptr || dstTypeInterface == nullptr) {
        return false;
    }

    const auto srcDistTypes = srcTypeInterface.getDistributedTypes();
    const auto dstDistTypes = dstTypeInterface.getDistributedTypes();
    if (srcDistTypes.size() != dstDistTypes.size() || srcDistTypes.empty()) {
        return false;
    }

    const auto srcDistType = srcDistTypes.front().cast<VPU::DistributedTensorType>();
    const auto dstDistType = dstDistTypes.front().cast<VPU::DistributedTensorType>();
    if (srcDistType.getShape() != dstDistType.getShape()) {
        return false;
    }
    if (srcDistType.getDimsOrder() != dstDistType.getDimsOrder() || srcDistType.getDimsOrder() != DimsOrder::NHWC) {
        return false;
    }

    const auto srcDistAttr = srcDistType.getDistribution();
    const auto dstDistAttr = dstDistType.getDistribution();
    if (srcDistAttr.getMode().getValue() != DistributionMode::SEGMENTED ||
        dstDistAttr.getMode().getValue() != DistributionMode::SEGMENTED) {
        return false;
    }
    if (srcDistAttr.getNumTiles() != dstDistAttr.getNumTiles()) {
        return false;
    }
    if ((srcDistType.getDistribution().getMode().getValue() != DistributionMode::SEGMENTED) ||
        (dstDistType.getDistribution().getMode().getValue() != DistributionMode::SEGMENTED) ||
        (srcDistType.getDistribution().getNumTiles() != dstDistType.getDistribution().getNumTiles())) {
        return false;
    }

    return true;
}

bool isTargetTensorTypeCompatible(vpux::NDTypeInterface srcType, vpux::NDTypeInterface dstType) {
    const auto srcTypeInterface = srcType.dyn_cast<VPU::DistributedTypeInterface>();
    const auto dstTypeInterface = dstType.dyn_cast<VPU::DistributedTypeInterface>();
    const auto srcTypeIsDistributed = srcTypeInterface != nullptr;
    const auto dstTypeIsDistributed = dstTypeInterface != nullptr;
    if (srcTypeIsDistributed ^ dstTypeIsDistributed) {
        return false;
    }
    if (srcTypeIsDistributed && dstTypeIsDistributed) {
        const auto srcContainsDistTypes = srcTypeInterface.containsDistributedTypes();
        const auto dstContainsDistTypes = dstTypeInterface.containsDistributedTypes();
        if (srcContainsDistTypes ^ dstContainsDistTypes) {
            return false;
        }
        if (srcContainsDistTypes && dstContainsDistTypes) {
            const auto distributedSrcTypes = srcTypeInterface.getDistributedTypes();
            const auto distributedDstTypes = dstTypeInterface.getDistributedTypes();
            if (distributedSrcTypes.size() != distributedDstTypes.size()) {
                return false;
            }
            for (auto p : zip(distributedSrcTypes, distributedDstTypes)) {
                const auto distributedSrcType = std::get<0>(p).cast<VPU::DistributedTensorType>();
                const auto distributedDstType = std::get<1>(p).cast<VPU::DistributedTensorType>();
                if (isDistributedCastCompatible(distributedSrcType, distributedDstType).failed()) {
                    return false;
                }
            }
        }
    }
    return true;
}

uint32_t getPrefetchDMACostOverlappsWithPreviousDPU(SmallVector<uint32_t>& layerDPUCosts,
                                                    ArrayRef<uint32_t> layerDMACosts, bool isDMAOverlapsWithDPU) {
    VPUX_THROW_UNLESS(layerDPUCosts.size() == layerDMACosts.size(), "Size of DPU and DMA costs should be equal.");
    VPUX_THROW_WHEN(layerDPUCosts.empty(), "DPU costs should not be empty.");

    uint32_t totalDMACost = 0;
    if (isDMAOverlapsWithDPU) {
        for (size_t tileIdx = 0; tileIdx < layerDMACosts.size() - 1; ++tileIdx) {
            if (layerDMACosts[tileIdx + 1] > layerDPUCosts[tileIdx]) {
                // Prefetched and all the DPU cycles are overlapped with DMA
                totalDMACost += (layerDMACosts[tileIdx + 1] - layerDPUCosts[tileIdx]);
                layerDPUCosts[tileIdx] = 0;
            } else {
                // Prefetched and some DPU cycles are still not overlapped
                layerDPUCosts[tileIdx] -= layerDMACosts[tileIdx + 1];
            }
        }
    }
    totalDMACost += layerDMACosts[0];
    return totalDMACost;
}

uint32_t getOutputDMACostOverlappsWithNextDPU(SmallVector<uint32_t>& layerDPUCosts, ArrayRef<uint32_t> layerDMACosts,
                                              bool isDMAOverlapsWithDPU) {
    VPUX_THROW_UNLESS(layerDPUCosts.size() == layerDMACosts.size(), "Size of DPU and DMA costs should be equal.");
    VPUX_THROW_WHEN(layerDPUCosts.empty(), "DPU costs should not be empty.");

    uint32_t totalDMACost = 0;
    if (isDMAOverlapsWithDPU) {
        for (size_t tileIdx = 0; tileIdx < layerDMACosts.size() - 1; ++tileIdx) {
            if (layerDMACosts[tileIdx] > layerDPUCosts[tileIdx + 1]) {
                // Prefetched and all the DPU cycles are overlapped with DMA
                totalDMACost += (layerDMACosts[tileIdx] - layerDPUCosts[tileIdx + 1]);
                layerDPUCosts[tileIdx + 1] = 0;
            } else {
                // Prefetched and some DPU cycles are still not overlapped
                layerDPUCosts[tileIdx + 1] -= layerDMACosts[tileIdx];
            }
        }
    }
    totalDMACost += layerDMACosts.back();
    return totalDMACost;
}

}  // namespace

LayerCostModel::LayerCostModel(mlir::func::FuncOp func, bool enablePrefetchTiling, Logger log)
        : _func(func), _enablePrefetchTiling(enablePrefetchTiling), _log(log) {
    auto module = func->getParentOfType<mlir::ModuleOp>();

    if (auto tileOp = IE::getTileExecutor(module)) {
        auto dpuExec = tileOp.getSubExecutor(VPU::ExecutorKind::DPU);
        _NCEFrequency = tileOp.getProcessorFrequency().getValueAsDouble();
        _numTiles = tileOp.getCount();
        _numDPUs = dpuExec.getCount();
        _NCEThroughput = getNCEThroughput(VPU::getArch(tileOp));
        _DMABandwidth = getDMABandwidth(VPU::getArch(tileOp), VPU::getRevisionID(module));
        if (auto shaveActExec = tileOp.getSubExecutor(ExecutorKind::SHAVE_ACT)) {
            _numShaveActs = shaveActExec.getCount();
        }
    }
    _numDMAPorts = IE::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN).getCount();
    _arch = VPU::getArch(module);
    _vpuDeviceType = VPU::getVPUDeviceType(_arch);
    _layerCostModel = VPU::createLayerCostModel(_arch);
}

vpux::NDTypeInterface LayerCostModel::getNormalInputType(VPU::ClusteredOpInterface origOp,
                                                         mlir::Operation* parentOp) const {
    auto input = getInputFromClusteredOp(origOp, parentOp);
    return input.getType().dyn_cast<vpux::NDTypeInterface>();
}

vpux::NDTypeInterface LayerCostModel::getNormalOutputType(VPU::ClusteredOpInterface origOp) const {
    auto output = origOp->getResult(0);
    return output.getType().dyn_cast<vpux::NDTypeInterface>();
}

VPU::DistributedTypeInterface LayerCostModel::getDistributedInputType(
        VPU::ClusteredOpInterface origOp, mlir::Operation* parentOp,
        VPU::MultiClusterStrategy specifiedStrategy) const {
    auto input = getInputFromClusteredOp(origOp, parentOp);
    auto numClustersAttr = VPU::getOptimalNumClusters(
            origOp, origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape(), specifiedStrategy);
    if (auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation())) {
        auto isFilter = nceOp->getNumOperands() > 1 && input == nceOp->getOperand(1) &&
                        !mlir::isa<VPU::NCEEltwiseOp>(origOp.getOperation());
        if (isFilter) {
            return getDistributedFilterTypeFromOp(nceOp, input.getType(), numClustersAttr, specifiedStrategy);
        }
    }
    return getDistributedActivationTypeFromOp(origOp, input.getType(), numClustersAttr, specifiedStrategy);
}

VPU::DistributedTypeInterface LayerCostModel::getDistributedInputType(VPU::ClusteredOpInterface origOp,
                                                                      mlir::Operation* parentOp,
                                                                      VPU::MultiClusterStrategy specifiedStrategy,
                                                                      mlir::ArrayAttr customAlignment) const {
    auto input = getInputFromClusteredOp(origOp, parentOp);
    auto numClustersAttr = VPU::getOptimalNumClusters(
            origOp, origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape(), specifiedStrategy);
    auto customAlignmentArr = customAlignment ? parseIntArrayAttr<int64_t>(customAlignment) : SmallVector<int64_t>{};
    return getDistributedActivationTypeFromOp(origOp, input.getType(), numClustersAttr, specifiedStrategy,
                                              customAlignmentArr);
}

VPU::DistributedTypeInterface LayerCostModel::getDistributedOutputType(
        VPU::ClusteredOpInterface origOp, VPU::MultiClusterStrategy specifiedStrategy) const {
    auto numClustersAttr = VPU::getOptimalNumClusters(
            origOp, origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape(), specifiedStrategy);
    return VPU::getDistributedOutputTypeFromOp(origOp, origOp->getResult(0).getType(), numClustersAttr,
                                               specifiedStrategy);
}

/*
 * Get the spilling cost
 * srcTensorType is the output of parent op (current op)
 * dstTensorType is the input of child op
 * return spilling write cost and spilling read cost
 */
LayerCostModel::SpillingCost LayerCostModel::getSpillingCost(vpux::NDTypeInterface srcTensorType,
                                                             vpux::NDTypeInterface dstTensorType,
                                                             VPU::ClusteredOpInterface parentOp,
                                                             VPU::ClusteredOpInterface userOp) const {
    // Concat is on DDR memory if there's spilling. So we don't need copy from CMX to DDR if Concat is parent. Also we
    // don't need copy from DDR to CMX if Concat is user.
    if (mlir::isa<VPU::ConcatOp>(parentOp)) {
        return {0.0, getSpillingReadCost(dstTensorType)};
    }

    if (mlir::isa<VPU::ConcatOp>(userOp)) {
        return {getSpillingWriteCost(srcTensorType), 0.0};
    }

    return {getSpillingWriteCost(srcTensorType), getSpillingReadCost(dstTensorType)};
}

double LayerCostModel::getDMACostOfType(vpux::NDTypeInterface srcType, SpillingType spillingType) const {
    auto distributedSrcType = srcType.dyn_cast<DistributedTensorType>();
    auto srcMode = distributedSrcType != nullptr ? distributedSrcType.getDistribution().getMode().getValue()
                                                 : VPU::DistributionMode::NONE;

    if (_arch == VPU::ArchKind::NPU37XX) {
        return static_cast<double>(getDMACost(srcType, _vpuDeviceType, _layerCostModel, _numDMAPorts));
    }

    auto spillingReadCostFunc = spillingCostMap.at(srcMode);
    return spillingReadCostFunc(srcType, spillingType, _DDRLatency, _DMABandwidth, _CMXLatency, _CMXMulticastBandwidth,
                                _numDMAPorts);
}

double LayerCostModel::getSpillingDMACost(vpux::NDTypeInterface srcTensorType, SpillingType spillingType) const {
    if (auto sparseTensorType = srcTensorType.dyn_cast<VPU::SparseTensorType>()) {
        srcTensorType = sparseTensorType.getData().cast<vpux::NDTypeInterface>();
    }
    return getDMACostOfType(srcTensorType, spillingType);
}

double LayerCostModel::getSpillingReadCost(vpux::NDTypeInterface srcTensorType) const {
    return getSpillingDMACost(srcTensorType, SpillingType::SPILL_READ);
}

double LayerCostModel::getSpillingWriteCost(vpux::NDTypeInterface srcTensorType) const {
    return getSpillingDMACost(srcTensorType, SpillingType::SPILL_WRITE);
}

// The function computes the actual output tensor volume (i.e. computation that is performed)
// given the stratey and the MPE mode
double LayerCostModel::calculateMPEVolume(VPU::MPEMode mpeMode, Shape shape) const {
    int64_t mpeHeight;
    int64_t mpeWidth;
    switch (mpeMode) {
    case VPU::MPEMode::VECTOR: {
        mpeHeight = 1;
        mpeWidth = 16;
        break;
    }
    case VPU::MPEMode::VECTOR_FP16: {
        mpeHeight = 1;
        mpeWidth = 4;
        break;
    }
    case VPU::MPEMode::MATRIX:
    // These different mpe modes on VPUX37XX have impact on the reuse of activation and weights. We can't estimate reuse
    // cost with current cost equation. In the future we will integrate VPUNN to estimate the cost.
    case VPU::MPEMode::CUBOID_4x16:
    case VPU::MPEMode::CUBOID_8x16:
    case VPU::MPEMode::CUBOID_16x16: {
        mpeHeight = 4;
        mpeWidth = 4;
        break;
    }
    default:
        VPUX_THROW("Unsupported mpeMode '{0}'", mpeMode);
    }

    return static_cast<double>(_numDPUs * divUp((mpeHeight * divUp(shape[Dims4D::Act::H], mpeHeight) * mpeWidth *
                                                 divUp(shape[Dims4D::Act::W], mpeWidth) * _numChannelAlignment *
                                                 divUp(shape[Dims4D::Act::C], _numChannelAlignment)),
                                                _numDPUs));
}

// The efficiency calculation that is being performed here can be described as follows.
// A ratio of the real output tensor volume to the actual computation that occurs on the
// hardware for each MPE Mode 4x4x16 and 16x1x16 is computed and the maximum is selected.
double LayerCostModel::computeSplitEfficiency(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp.checkStrategyCompatibility(strategy, _numTiles) == true,
                      "Unsupported multi-cluster strategy '{0}' for layer type '{1}'", strategy, nceOp->getName());
    auto numClusters = getOptimalNumClusters(
            clusteredOp, nceOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getShape(), strategy);
    const auto distributedOutputTensorType = VPU::getDistributedOutputTypeFromOp(
            clusteredOp, nceOp->getResult(0).getType().cast<vpux::NDTypeInterface>(), numClusters, strategy);

    VPUX_THROW_UNLESS(distributedOutputTensorType.containsDistributedTypes(), "Missing output distributed types");
    const auto distributedOutputDataType =
            distributedOutputTensorType.getDistributedTypes().front().cast<VPU::DistributedTensorType>();
    const auto perClusterShape = distributedOutputDataType.getLargestCompactShape();
    const auto perClusterOutputTensorVolume =
            perClusterShape[Dims4D::Act::H] * perClusterShape[Dims4D::Act::W] * perClusterShape[Dims4D::Act::C];

    const auto arch = VPU::getArch(nceOp);

    // VPUX37XX has different kinds of MPE mode
    if (arch == VPU::ArchKind::NPU37XX) {
        return std::max(std::max(static_cast<double>(perClusterOutputTensorVolume) /
                                         calculateMPEVolume(VPU::MPEMode::CUBOID_4x16, perClusterShape),
                                 static_cast<double>(perClusterOutputTensorVolume) /
                                         calculateMPEVolume(VPU::MPEMode::CUBOID_8x16, perClusterShape)),
                        static_cast<double>(perClusterOutputTensorVolume) /
                                calculateMPEVolume(VPU::MPEMode::CUBOID_16x16, perClusterShape));
    } else {
        return std::max(static_cast<double>(perClusterOutputTensorVolume) /
                                calculateMPEVolume(VPU::MPEMode::MATRIX, perClusterShape),
                        static_cast<double>(perClusterOutputTensorVolume) /
                                calculateMPEVolume(VPU::MPEMode::VECTOR, perClusterShape));
    }
}

// Returns the duration in cycles for the execution of a NCE task
double LayerCostModel::clusterComputeTime(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp.checkStrategyCompatibility(strategy, _numTiles) == true,
                      "Unsupported multi-cluster strategy '{0}' for layer type '{1}'", strategy, nceOp->getName());

    double clusterOpsPerCycle = _NCEThroughput / _NCEFrequency / _numTiles;
    double clusterEff = computeSplitEfficiency(nceOp, strategy);
    auto largestClusterOutShape = getLargestClusterOutputShape(clusteredOp, strategy);

    auto kernelSize = nceOp.getKernelSizeVal();
    auto op = nceOp.getOperation();
    int64_t baseKernelCost = kernelSize[Dims4D::Kernel::Y.ind()] * kernelSize[Dims4D::Kernel::X.ind()];
    if (mlir::isa<VPU::NCEConvolutionOp, VPU::NCECompressConvolutionOp, VPU::NCEInterpolateOp>(op)) {
        int64_t IC = getShape(
                op->getOperand(0))[Dims4D::Act::C];  // Get input channel (already channel-alignment in previous pass)
        baseKernelCost = IC * baseKernelCost;

    } else if (mlir::isa<VPU::NCEEltwiseOp>(op)) {
        baseKernelCost = 1;
    } else if (!mlir::isa<VPU::NCEMaxPoolOp>(op) && !mlir::isa<VPU::NCEAveragePoolOp>(op) &&
               !mlir::isa<VPU::NCEDepthConvolutionOp>(op)) {
        VPUX_THROW("Invalid NCE operation type: '{0}'", op->getName());
    }

    // Here calculating the total basic operation number for the largest cluster output
    // And also we can reduce formula like:
    // basicOperationVolume = clusterOutShapeSize * baseKernelCost / Efficiency
    //                      = clusterOutShapeSize * baseKernelCost / (clusterOutShapeSize / MPEVolume) =
    //                      = MPEVolume * baseKernelCost
    // So that MPEVolume * baseKernelCost is the final result, and then we can divide frequency to get final cycles
    double basicOperationVolume =
            (static_cast<double>(largestClusterOutShape.totalSize() * baseKernelCost)) / clusterEff;
    double clusterComputeCycles = basicOperationVolume / clusterOpsPerCycle;
    return clusterComputeCycles;
}

/// @brief Returns total number of cycles required to weights DMA and CMX broadcast
/// for a layer with given strategy
/// @details Data transferring cost is modeled as (latency + size / bandwidth)
double LayerCostModel::totalDMATime(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp.checkStrategyCompatibility(strategy, _numTiles) == true,
                      "Unsupported multi-cluster strategy '{0}' for layer type '{1}'", strategy, nceOp->getName());
    double totalActivationWindowCycles = 0.0;
    double totalWeightCycles = 0.0;
    double totalWeightsTableCycles = 0.0;
    double pwlTableCycles = 0.0;
    double outputCycles = 0.0;
    const auto op = nceOp.getOperation();
    const int64_t IC = (mlir::isa<VPU::NCEConvolutionOp, VPU::NCECompressConvolutionOp, VPU::NCEInterpolateOp>(op))
                               ? getShape(op->getOperand(0))[Dims4D::Act::C]
                               : 1;
    const int64_t OC = getShape(op->getResult(0))[Dims4D::Act::C];
    const auto kernelSize = nceOp.getKernelSizeVal();
    auto numClustersAttr = VPU::getOptimalNumClusters(clusteredOp, getShape(op->getResult(0)), strategy);

    /// Weights cost
    /// Weights and weightTable are Segmented mode under SOK (only including ddr -> cmx cost),
    /// SOK may use less clusters to avoid alignment
    /// So it's not proper to estimate total weightsSize by "clusterWeightsSize * _numTiles" simply
    /// Using distributed tensor for SOK to get accurate total size
    if (mlir::isa<VPU::NCEConvolutionOp, VPU::NCEDepthConvolutionOp, VPU::NCECompressConvolutionOp,
                  VPU::NCEInterpolateOp>(op)) {
        auto weights = op->getOperand(1);

        SmallVector<int64_t> numElemsPerOC;
        int64_t weightSetAlignment = _cmxAddressAlignment;
        if (auto sparseWeightsType = weights.getType().dyn_cast<VPU::SparseTensorType>()) {
            VPU::SparsityCompressionAttr sparsityCompression = sparseWeightsType.getSparsityCompression();
            if (sparsityCompression != nullptr && sparsityCompression.getAxis() != nullptr) {
                auto axis = sparsityCompression.getAxis().getInt();
                VPUX_THROW_UNLESS(axis == Dims4D::Filter::OC.ind(),
                                  "SplitOverK is only compatible with compression over OC");
                numElemsPerOC = to_small_vector(sparsityCompression.getNumElems().getValues<int64_t>());
                if (sparsityCompression.getAlignment() != nullptr) {
                    weightSetAlignment = sparsityCompression.getAlignment().getInt();
                }

                auto weightsShape = sparseWeightsType.getShape();
                VPUX_THROW_UNLESS(
                        static_cast<int64_t>(numElemsPerOC.size()) == weightsShape[Dims4D::Filter::OC],
                        "Different number of output channels {0} compared to compression scheme with {1} elements",
                        weightsShape[Dims4D::Filter::OC], numElemsPerOC.size());
            }
        }

        if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
            int64_t totalWeightsSize = 0;
            auto distributedWeightsTensorType =
                    VPU::getDistributedFilterTypeFromOp(nceOp, weights.getType(), numClustersAttr, strategy);

            for (auto type : distributedWeightsTensorType.getDistributedTypes() | indexed) {
                auto distributedWeightsType = type.value().cast<VPU::DistributedTensorType>();
                auto tiledWeightsShapes = distributedWeightsType.getPerClusterMemoryShapes();
                auto tiledWeightsOffsets = distributedWeightsType.getPerClusterMemoryShapeOffsets();

                const Bit elemBitSize = getElemTypeSize(distributedWeightsType);
                int64_t weightsByteSize = 0;

                if (type.index() == 0 && !numElemsPerOC.empty()) {
                    for (auto p : zip(tiledWeightsShapes, tiledWeightsOffsets)) {
                        const auto tileShape = std::get<0>(p);
                        const auto tileOffsets = std::get<1>(p);
                        const auto startOC = tileOffsets[Dims4D::Filter::OC];
                        const auto endOC = startOC + tileShape[Dims4D::Filter::OC];
                        for (auto idx = startOC; idx < endOC; ++idx) {
                            const auto numElems = *(numElemsPerOC.begin() + idx);
                            weightsByteSize +=
                                    alignMemSize(elemBitSize * numElems, Byte(weightSetAlignment)).to<Byte>().count();
                        }
                    }
                } else {
                    for (auto& clusterWeightsShape : tiledWeightsShapes) {
                        weightsByteSize +=
                                alignMemSize(elemBitSize * clusterWeightsShape.totalSize(), Byte(1)).to<Byte>().count();
                    }
                }

                totalWeightsSize += weightsByteSize;
            }

            const double weightCycles = static_cast<double>(totalWeightsSize) / _DMABandwidth;
            totalWeightCycles = _DDRLatency + weightCycles;
        } else {
            // Duplicated mode in other strategies has ddr->cmx cost
            // The weight set size (IC * KX * KY * BytesPerElement) needs to be aligned to 16B for kernels

            auto ndWeightsType = weights.getType().cast<vpux::NDTypeInterface>();
            const Bit elemBiSize = ndWeightsType.getElemTypeSize();
            const int64_t weightSetSize =
                    IC * kernelSize[Dims4D::Kernel::Y.ind()] * kernelSize[Dims4D::Kernel::X.ind()];

            int64_t clusterWeightsSize = 0;
            if (!numElemsPerOC.empty()) {
                for (auto numElems : numElemsPerOC) {
                    clusterWeightsSize +=
                            alignMemSize(elemBiSize * numElems, Byte(_cmxAddressAlignment)).to<Byte>().count();
                }
            } else {
                clusterWeightsSize +=
                        OC * alignMemSize(elemBiSize * weightSetSize, Byte(_cmxAddressAlignment)).to<Byte>().count();
            }

            if (weights.getType().isa<VPU::SparseTensorType>()) {
                const int64_t weightSetBitAlignment = 128;
                const int64_t sparsityMapSize = (OC * alignValUp<int64_t>(weightSetSize, weightSetBitAlignment));
                const int64_t sparsityMapByteSize = sparsityMapSize / CHAR_BIT;
                clusterWeightsSize += sparsityMapByteSize;
            }

            const double weightCycles = static_cast<double>(clusterWeightsSize) / _DMABandwidth;
            totalWeightCycles = _DDRLatency + weightCycles;
        }
    }

    /// WeightsTable cost
    /// WeightTable has OC entries, each entry includes sparsity/weights pointer, bias and mult/shfit quantized
    /// params. The total size for each entry is 16 Bytes
    if (nceOp.getWeightsTableOperand() != nullptr) {
        auto largestClusterOutShape = getLargestClusterOutputShape(clusteredOp, strategy);
        int64_t alignedClusterOutChannels = largestClusterOutShape[Dims4D::Act::C];
        int64_t clusterWeightTableSize = NCEInvariant::getWeightsTableSize(alignedClusterOutChannels).count();

        if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
            totalWeightsTableCycles =
                    _DDRLatency +
                    static_cast<double>(clusterWeightTableSize * numClustersAttr.getInt()) / _DMABandwidth;
        } else {
            totalWeightsTableCycles = _DDRLatency + (static_cast<double>(clusterWeightTableSize) / _DMABandwidth);
        }
    }

    /// ActivationWindow cost
    /// It's always duplicated mode and only dwconv , cmconv and maxpool own it
    if (nceOp.getActivationWindowOperand() != nullptr) {
        const auto SX = Shape(nceOp.getStridesVal())[Dims4D::Strides::X];
        const auto inputElemType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>().getElementType();
        auto sparsityMode = VPU::NCESparsity::Mode::POOL;
        if (mlir::isa<VPU::NCEConvolutionOp>(op)) {
            sparsityMode = VPU::NCESparsity::Mode::CM_CONV;
        } else if (mlir::isa<VPU::NCEDepthConvolutionOp>(op)) {
            sparsityMode = VPU::NCESparsity::Mode::DW_CONV;
        }

        const auto activationWindowSize =
                VPU::NCESparsity::getActivationWindowSize(sparsityMode, Shape(kernelSize), SX, inputElemType, IC);
        totalActivationWindowCycles = _DDRLatency + (static_cast<double>(activationWindowSize) / _DMABandwidth);
    }

    /// PWLTable cost
    /// DistributionMode is Duplicated, at most we have 32 instructions (8 segments for pwl),
    /// so size is 32 x INT32 = 128 Bytes
    if (auto origOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(op)) {
        if (origOp.getInstructionListTable()) {
            pwlTableCycles = _DDRLatency + (128.0 / _DMABandwidth);
        }
    } else if (auto origOp = mlir::dyn_cast<VPU::NCEDepthConvolutionOp>(op)) {
        if (origOp.getInstructionListTable()) {
            pwlTableCycles = _DDRLatency + (128.0 / _DMABandwidth);
        }
    }

    // Total cost for single layer
    return totalActivationWindowCycles + totalWeightCycles + totalWeightsTableCycles + pwlTableCycles + outputCycles;
}

/// @brief A switcher to select time-cost or efficiency-cost for greedy strategy assignment
/// @details Time-cost includes an extra input spilling cost to be more accurate
double LayerCostModel::getNCELayerCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy,
                                       bool useTimeBasedCost) {
    if (!useTimeBasedCost) {
        return getEfficiencyCost(nceOp, strategy);
    }

    double basicDPUandDMACost = COST_MAX;

    const auto it = _costCache.find(nceOp);
    if (it == _costCache.end()) {
        // Case 1 - Op costs are not found in cache:
        // 1.Calculate cost value with VPUNN
        // 2.Create new op costs
        // 3.Store the new op costs in cost cache
        basicDPUandDMACost = getDPUandDMATimeCost(nceOp, strategy);
        SmallVector newOpCosts((getMaxEnumValForMultiClusterStrategy() + 1), COST_MAX);
        newOpCosts[static_cast<uint64_t>(strategy)] = basicDPUandDMACost;
        _costCache.insert({nceOp, newOpCosts});
    } else {
        auto strategyCostIt = it->second.begin() + static_cast<uint64_t>(strategy);
        if (strategyCostIt != nullptr && *strategyCostIt != COST_MAX) {
            // Case 2 - Strategy cost is found in cache:
            // Retrieve the cost value directly
            basicDPUandDMACost = *strategyCostIt;
        } else {
            // Case 3 - Op costs are found but op strategy cost is not found:
            // 1.Calculate cost value with VPUNN
            // 2.Update op strategy cost value in cache
            basicDPUandDMACost = getDPUandDMATimeCost(nceOp, strategy);
            *strategyCostIt = basicDPUandDMACost;
        }
    }

    return basicDPUandDMACost;
}

/// @brief Time-cost : return the shave computation time (cycles)
/// @details use vpunn cost model to get the shave cost of sw layer
double LayerCostModel::getSWLayerCost(VPU::SWOpInterface swOp, VPU::MultiClusterStrategy strategy) const {
    auto getVPUTensors = [&](mlir::ValueRange values) -> std::vector<VPUNN::VPUTensor> {
        std::vector<VPUNN::VPUTensor> tensors;
        std::transform(values.begin(), values.end(), std::back_inserter(tensors), [](mlir::Value value) {
            auto valueType = value.getType().cast<vpux::NDTypeInterface>();
            return VPU::getVPUTensor(valueType.getShape(), valueType.getElementType());
        });
        return tensors;
    };

    const auto device = VPU::getVPUDeviceType(_arch);
    const auto inputTensors = getVPUTensors(swOp->getOperands());
    const auto outputTensors = getVPUTensors(swOp->getResults());

    std::shared_ptr<VPUNN::SWOperation> vpunnLayer;
    llvm::TypeSwitch<mlir::Operation*, void>(swOp.getOperation())
            .Case<VPU::TanhOp>([&](VPU::TanhOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVTanh>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::MVNOp>([&](VPU::MVNOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVMVN>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::SoftMaxOp>([&](VPU::SoftMaxOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVSoftmax>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::SwishOp>([&](VPU::SwishOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVSwish>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::HSwishOp>([&](VPU::HSwishOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVHardSwish>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::FloorOp>([&](VPU::FloorOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVFloor>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::RoundOp>([&](VPU::RoundOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVRound>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::SinOp>([&](VPU::SinOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVSin>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::CosOp>([&](VPU::CosOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVCos>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::ExpOp>([&](VPU::ExpOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVExp>(device, inputTensors.front(), outputTensors.front());
            })
            .Case<VPU::PowerOp>([&](VPU::PowerOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVPower>(device, inputTensors, outputTensors.front());
            })
            .Case<VPU::DivideOp>([&](VPU::DivideOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVDivide>(device, inputTensors, outputTensors.front());
            })
            .Case<VPU::GreaterOp>([&](VPU::GreaterOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVGreater>(device, inputTensors, outputTensors.front());
            })
            .Case<VPU::LessOp>([&](VPU::LessOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVLess>(device, inputTensors, outputTensors.front());
            })
            .Case<VPU::EqualOp>([&](VPU::EqualOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVEqual>(device, inputTensors, outputTensors.front());
            })
            .Case<VPU::AndOp>([&](VPU::AndOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVAnd>(device, inputTensors, outputTensors.front());
            })
            .Case<VPU::SubtractOp>([&](VPU::SubtractOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVSubtract>(device, inputTensors, outputTensors.front());
            })
            .Case<VPU::AddOp>([&](VPU::AddOp) {
                vpunnLayer = std::make_shared<VPUNN::SHVAdd>(device, inputTensors, outputTensors.front());
            })
            .Default([&](mlir::Operation* op) {
                VPUX_THROW("SW op {0} has no VPUNN support", op->getName());
            });
    auto vpunnStrategy = VPU::getVPULayerStrategy(strategy, _numDPUs, _numTiles, _numShaveActs, false);
    return _layerCostModel->Layer(*vpunnLayer, vpunnStrategy);
}

// Predict the cost considering overlapping with the parent compute time
// The cost is for reference and can not be completely accurate compared to the profiling
double LayerCostModel::getTheLastTileComputeCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy opStrategy) {
    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(nceOp.getOperation());
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "NCE op {0} is not a TilingBuilderOpInterface");
    OutputTiling tileInfo({TileInfo(getShape(nceOp->getResult(0)))});
    if (opNeedsTiling(nceOp.getOperation(), true, _log)) {
        const auto tilingStrategy = getLayerTilingStrategy(tilingBuilder, true, _log);
        VPUX_THROW_UNLESS(mlir::succeeded(tilingStrategy), "Can not get tiling strategy for op at '{0}'",
                          nceOp.getLoc());
        tileInfo = tilingStrategy.value();
    }
    SmallVector<SmallVector<NDTypeInterface>> tilesTypes;
    const auto costParams = VPU::getWorkloadCostParam(nceOp, _arch, _numDPUs);
    const auto vpunnStrategy = VPU::getVPULayerStrategy(opStrategy, _numDPUs, _numTiles, _numShaveActs, true);
    auto DPUCosts = getDPUCostForNCEOp(nceOp, opStrategy, tileInfo, tilesTypes, costParams, vpunnStrategy,
                                       _layerCostModel, _log);
    if (DPUCosts.empty()) {
        _log.warning("Can not get DPU cost for NCE op {0}", nceOp->getName());
        return 0;
    }
    return DPUCosts.back();
}

/// @brief get computation cost
/// @details Time-cost includes an extra input spilling cost to be more accurate
double LayerCostModel::getLayerCost(mlir::Operation* op, VPU::MultiClusterStrategy strategy, bool useTimeBasedCost,
                                    bool onlyLastComputeCost) {
    return llvm::TypeSwitch<mlir::Operation*, double>(op)
            .Case<VPU::NCEOpInterface>([&](VPU::NCEOpInterface nceOp) -> double {
                if (onlyLastComputeCost) {
                    // Return the compute cost of the parent op
                    // If the parent op has tiling strategy, return the compute cost of the last tile
                    return getTheLastTileComputeCost(nceOp, strategy);
                }
                return getNCELayerCost(nceOp, strategy, useTimeBasedCost);
            })
            .Case<VPU::SWOpInterface>([&](VPU::SWOpInterface swOp) -> double {
                if (!swOp.supportCycleCostCalculation()) {
                    return 0;
                }
                return getSWLayerCost(swOp, strategy);
            })
            .Default([](mlir::Operation*) -> double {
                return 0;
            });
}

double LayerCostModel::getDPUandDMATimeCostWithCustomTiling(VPU::NCEOpInterface nceOp,
                                                            VPU::MultiClusterStrategy strategy,
                                                            const OutputTiling& outTiles) {
    // Types for each tile
    SmallVector<SmallVector<NDTypeInterface>> tilesTypes;

    _log.trace("Start calculating VPUNN layer cost for Op {0} with strategy {1}", nceOp.getLoc(), strategy);

    const auto costParams = VPU::getWorkloadCostParam(nceOp, _arch, _numDPUs);
    const auto vpunnStrategy = VPU::getVPULayerStrategy(strategy, _numDPUs, _numTiles, 1, true);
    auto vpunnLayerDPUCosts =
            getDPUCostForNCEOp(nceOp, strategy, outTiles, tilesTypes, costParams, vpunnStrategy, _layerCostModel, _log);
    if (vpunnLayerDPUCosts.empty()) {
        return COST_MAX;
    }
    _log.trace("VPUNN DPU layer costs {0}", vpunnLayerDPUCosts);

    const auto getSpillingReadCost = [&](NDTypeInterface srcType) -> uint32_t {
        return checked_cast<uint32_t>(this->getSpillingReadCost(srcType));
    };

    const auto getSpillingWriteCost = [&](NDTypeInterface srcType) -> uint32_t {
        return checked_cast<uint32_t>(this->getSpillingWriteCost(srcType));
    };

    double cost = 0;

    // Accumulate all the DPU costs
    cost += std::accumulate(vpunnLayerDPUCosts.begin(), vpunnLayerDPUCosts.end(), 0);

    // Add weights DMA costs
    auto vpunnLayerWeightsCosts = getPerTileWeightsDMACosts(nceOp, tilesTypes, getSpillingReadCost);
    _log.trace("VPUNN weights DMA costs {0}", vpunnLayerWeightsCosts);
    auto getParentLastTileComputeCost = [&]() -> double {
        // Predict the cost considering overlapping with the parent compute time
        // The cost is for reference and can not be completely accurate compared to the profiling
        auto parentOp = getParentComputeOp(nceOp.getOperation());
        if (parentOp == nullptr || !mlir::isa<VPU::ClusteredOpInterface>(parentOp)) {
            return 0;
        }
        if (mlir::isa<VPU::NCEPermuteOp>(parentOp)) {
            // Skip NCEPermute parent because of inaccurate VPUNN cost: ticket E#89715
            return MAX_VAL;
        }
        auto parentOpStrategy = hasMultiClusterStrategy(parentOp)
                                        ? getMultiClusterStrategyValue(mlir::cast<VPU::ClusteredOpInterface>(parentOp))
                                        : VPU::MultiClusterStrategy::Clustering;

        return getLayerCost(parentOp, parentOpStrategy, /*useTimeBasedCost*/ true, /*onlyLastComputeCost*/ true);
    };
    // The first weights DMA should not be counted, overlapping with the parent op
    auto tilingInfoOp = mlir::dyn_cast<VPU::TilingInfoOpInterface>(nceOp.getOperation());
    const auto outShape = getShape(nceOp->getResult(0));
    auto tiles = outTiles.empty() ? OutputTiling({TileInfo(outShape)}) : outTiles;
    bool isFirstWeightsDMAOverlappedWithParent =
            _enablePrefetchTiling ? tilingInfoOp != nullptr &&
                                            tilingInfoOp.isSupportedTiling(tiles, vpux::TilingMode::PREFETCHING, _log)
                                  : false;
    const auto parentLastComputeCost =
            isFirstWeightsDMAOverlappedWithParent ? checked_cast<uint32_t>(getParentLastTileComputeCost()) : 0;
    _log.trace("VPUNN parent cost {0}", parentLastComputeCost);
    cost += getWeightsDMACostForNCEOp(nceOp, outTiles, vpunnLayerDPUCosts, vpunnLayerWeightsCosts,
                                      parentLastComputeCost, _enablePrefetchTiling, _log);

    // Add activation DMA costs
    auto vpunnLayerActCosts = getPerTileActivationDMACosts(nceOp, tilesTypes, getSpillingReadCost);
    _log.trace("VPUNN activation DMA costs {0}", vpunnLayerActCosts);
    cost += getActivationDMACostForNCEOp(nceOp, outTiles, vpunnLayerDPUCosts, vpunnLayerActCosts, _enablePrefetchTiling,
                                         _log);

    // Add output spilling cost
    // for non clusteredOp, must be ops that requires tiling
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
    if (clusteredOp == nullptr || !clusteredOp.doesLayerFitIntoCMX(strategy, /*reservedMem=*/Byte(0))) {
        // Consider output spilling pipelining with the next tile's DPU
        // Might be inaccurate when the DPU time is smaller than the sum of DMA time (input + weights + output)
        auto vpunnLayerOutputCosts = getPerTileOutputDMACosts(nceOp, tilesTypes, getSpillingWriteCost);
        _log.trace("VPUNN output DMA costs {0}", vpunnLayerOutputCosts);
        cost += getOutputDMACostForNCEOp(nceOp, outTiles, vpunnLayerDPUCosts, vpunnLayerOutputCosts,
                                         _enablePrefetchTiling, _log);
    }

    return cost;
}

/// @brief Time-cost : return a sum of layer DPU time and weights DMA time (cycles)
/// @details DPU time calculation also considers the impact of workloads split efficiency
double LayerCostModel::getDPUandDMATimeCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) {
    if (_arch == VPU::ArchKind::NPU37XX || _arch == VPU::ArchKind::NPU40XX) {
        auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
        VPUX_THROW_WHEN(clusteredOp == nullptr, "NCE op {0} at {1} should be a clustered op", nceOp->getName(),
                        nceOp.getLoc());

        // Set customized strategy to the op to get corresponding output tiles when tiling
        // Save and restore original strategy if needed
        auto origStrategy = clusteredOp.getMultiClusterStrategy();
        clusteredOp.setMultiClusterStrategy(strategy);

        // Output tiling for each tile
        OutputTiling outTiles({TileInfo(getShape(nceOp->getResult(0)))});

        // Check CMX memory as VPUNN works with layer which fits CMX memory
        // If not, tiling big layer to fit into CMX
        if (!(clusteredOp.doesLayerFitIntoCMX(strategy, /*reservedMem=*/Byte(0)))) {
            _log.trace("Tiling op {0} to fit into cmx before passing to VPUNN Layer API", nceOp.getLoc());
            auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(nceOp.getOperation());
            VPUX_THROW_WHEN(tilingBuilderOp == nullptr, "NCE op {0} at {1} should be a tiling op", nceOp->getName(),
                            nceOp.getLoc());

            auto tiles = getLayerTilingStrategy(tilingBuilderOp, _enablePrefetchTiling, _log);
            if (mlir::failed(tiles)) {
                _log.trace("Invalid tiling strategy for {0}", nceOp->getName());
                return COST_MAX;
            }
            outTiles = tiles.value();
        }

        auto cost = getDPUandDMATimeCostWithCustomTiling(nceOp, strategy, outTiles);

        _log.trace("VPUNN total layer cost for {0} is {1}", strategy, cost);

        // Restore original strategy or remove temporary strategy
        if (origStrategy.has_value()) {
            clusteredOp.setMultiClusterStrategy(origStrategy.value());
        } else {
            clusteredOp->removeAttr(multiClusterStrategy);
        }

        return cost;
    }

    // For KMB
    return clusterComputeTime(nceOp, strategy) + totalDMATime(nceOp, strategy);
}

///@brief Effi-cost : A simple cost considering DPU computing efficiency
double LayerCostModel::getEfficiencyCost(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy strategy) const {
    return 1.0 / computeSplitEfficiency(nceOp, strategy);
}

bool LayerCostModel::hasMultiClusterStrategy(mlir::Operation* op) const {
    if (auto clusteringOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op)) {
        return clusteringOp.getMultiClusterStrategy().has_value();
    }

    return false;
}

VPU::MultiClusterStrategy LayerCostModel::getMultiClusterStrategyValue(VPU::ClusteredOpInterface clusteredOp) const {
    auto strategy = clusteredOp.getMultiClusterStrategy();
    if (!strategy.has_value()) {
        VPUX_THROW("NCE operation {0} doesn't have multiClusterStrategy attribute", clusteredOp->getLoc());
    }

    return strategy.value();
}

/// Anywhere if you need to judge spilling existing, please call me!
/// srcTensorType is the output of parent origOp
/// dstTensorType is the input of child NCE op
bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface /*clusteredOp*/, vpux::NDTypeInterface srcTensorType,
                                 vpux::NDTypeInterface dstTensorType) const {
    auto getActivationTypeFromSparseType = [](vpux::NDTypeInterface tensorType) {
        if (auto sparseTensorType = tensorType.dyn_cast<VPU::SparseTensorType>()) {
            // interested in activation spills so use data for compatibility
            return sparseTensorType.getData().dyn_cast<vpux::NDTypeInterface>();
        }
        return tensorType;
    };

    srcTensorType = getActivationTypeFromSparseType(srcTensorType);
    dstTensorType = getActivationTypeFromSparseType(dstTensorType);

    if (isTargetTensorTypeCompatible(srcTensorType, dstTensorType) ||
        isSOHAlignmentCompatibleOrAdjustedCompatible(srcTensorType, dstTensorType)) {
        return false;
    }
    return true;
}

std::pair<vpux::NDTypeInterface, vpux::NDTypeInterface> LayerCostModel::getDistributionTypesWithStrategy(
        VPU::ClusteredOpInterface parentOp, VPU::MultiClusterStrategy parentStrategy, VPU::ClusteredOpInterface userOp,
        VPU::MultiClusterStrategy userStrategy) const {
    // Set the custom strategy to the op to get the accurate distributed type
    // The distribution mode depends on the neighboring op's strategy
    // e.g., Conv (SOK) -> SW (SOK), the output of the Conv would be SEGMENTED
    // Conv (SOK) -> SW (Clustering), the output of the Conv would be DUPLICATED|SEGMENTED
    // The DistributedType is decided by the ops multiCluster strategy attributes
    auto greedyStrategyParentOp = getMultiClusterStrategyValue(parentOp);
    auto greedyStrategyUserOp = getMultiClusterStrategyValue(userOp);
    parentOp.setMultiClusterStrategy(parentStrategy);
    userOp.setMultiClusterStrategy(userStrategy);
    auto targetOutputType = getDistributedOutputType(parentOp, parentStrategy);
    auto targetInputType = getDistributedInputType(userOp, parentOp, userStrategy);
    parentOp.setMultiClusterStrategy(greedyStrategyParentOp);
    userOp.setMultiClusterStrategy(greedyStrategyUserOp);

    // Adjust inputType alignment for SW op
    // e.g., Conv (SOK) -> SW (SOK), the input of SW can have a same alignment with Conv
    // to avoid spilling
    auto parentOutputDistType = targetOutputType.dyn_cast<VPU::DistributedTensorType>();
    auto userInputDistType = targetInputType.dyn_cast<VPU::DistributedTensorType>();
    if (parentOutputDistType != nullptr && userInputDistType != nullptr) {
        auto parentOutAlignment = parentOutputDistType.getDistribution().getAlignment();
        auto UserInAlignment = userInputDistType.getDistribution().getAlignment();
        if (parentOutAlignment != nullptr && UserInAlignment == nullptr &&
            mlir::isa<VPU::SWOpInterface>(userOp.getOperation()) &&
            isSWOpChannelAlignmentCompatible(userOp, targetInputType,
                                             userOp->getResult(0).getType().cast<vpux::NDTypeInterface>())) {
            targetInputType = getDistributedInputType(userOp, parentOp, userStrategy, parentOutAlignment);
        }
    }
    return {targetOutputType, targetInputType};
}

bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface clustered, VPU::ClusteredOpInterface userOp) const {
    auto targetOutputType = hasMultiClusterStrategy(clustered)
                                    ? getDistributedOutputType(clustered, getMultiClusterStrategyValue(clustered))
                                              .cast<vpux::NDTypeInterface>()
                                    : getNormalOutputType(clustered);

    auto targetInputType = hasMultiClusterStrategy(userOp)
                                   ? getDistributedInputType(userOp, clustered, getMultiClusterStrategyValue(userOp))
                                             .cast<vpux::NDTypeInterface>()
                                   : getNormalInputType(userOp, clustered);
    return hasSpilling(clustered, targetOutputType, targetInputType);
}

bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface origOp, VPU::MultiClusterStrategy origOpStrategy,
                                 VPU::ClusteredOpInterface userOp) const {
    auto targetOutputType = getDistributedOutputType(origOp, origOpStrategy).cast<vpux::NDTypeInterface>();
    auto targetInputType = hasMultiClusterStrategy(userOp)
                                   ? getDistributedInputType(userOp, origOp, getMultiClusterStrategyValue(origOp))
                                             .cast<vpux::NDTypeInterface>()
                                   : getNormalInputType(userOp, origOp);
    if (hasMultiClusterStrategy(userOp)) {
        std::tie(targetOutputType, targetInputType) =
                getDistributionTypesWithStrategy(origOp, origOpStrategy, userOp, getMultiClusterStrategyValue(userOp));
    }
    return hasSpilling(origOp, targetOutputType, targetInputType);
}

bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface origOp, VPU::ClusteredOpInterface userOp,
                                 VPU::MultiClusterStrategy userOpStrategy) const {
    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    auto targetOutputType = hasMultiClusterStrategy(origOp)
                                    ? getDistributedOutputType(origOp, getMultiClusterStrategyValue(clusteredOp))
                                              .cast<vpux::NDTypeInterface>()
                                    : getNormalOutputType(origOp);
    auto targetInputType = getDistributedInputType(userOp, origOp, userOpStrategy).cast<vpux::NDTypeInterface>();
    if (hasMultiClusterStrategy(origOp)) {
        std::tie(targetOutputType, targetInputType) =
                getDistributionTypesWithStrategy(origOp, getMultiClusterStrategyValue(origOp), userOp, userOpStrategy);
    }
    return hasSpilling(origOp, targetOutputType, targetInputType);
}

bool LayerCostModel::hasSpilling(VPU::ClusteredOpInterface origOp, VPU::MultiClusterStrategy origOpStrategy,
                                 VPU::ClusteredOpInterface userOp, VPU::MultiClusterStrategy userOpStrategy) const {
    auto targetTypes = getDistributionTypesWithStrategy(origOp, origOpStrategy, userOp, userOpStrategy);
    auto targetOutputType = targetTypes.first;
    auto targetInputType = targetTypes.second;
    return hasSpilling(origOp, targetOutputType, targetInputType);
}

bool LayerCostModel::doesLayerRequireTiling(VPU::ClusteredOpInterface clusteredOp,
                                            VPU::MultiClusterStrategy strategy) const {
    return !(clusteredOp.doesLayerFitIntoCMX(strategy, /*reservedMem=*/Byte(0)));
}

bool LayerCostModel::doesLayerHaveVPUNNSupportedTypes(VPU::ClusteredOpInterface clusteredOp) const {
    const bool hasSupportedOperandTypes = llvm::all_of(clusteredOp->getOperands(), [](const mlir::Value val) {
        return vpux::VPU::isVPUNNSupportedElementType(val.getType().cast<vpux::NDTypeInterface>().getElementType());
    });
    const bool hasSupportedResultTypes = llvm::all_of(clusteredOp->getResults(), [](const mlir::Value val) {
        return vpux::VPU::isVPUNNSupportedElementType(val.getType().cast<vpux::NDTypeInterface>().getElementType());
    });
    return hasSupportedOperandTypes && hasSupportedResultTypes;
}

LayerCostModel::SpillingCost LayerCostModel::calculateSpillingCost(VPU::ClusteredOpInterface parentOp,
                                                                   VPU::ClusteredOpInterface userOp,
                                                                   VPU::MultiClusterStrategy parentStrategy,
                                                                   VPU::MultiClusterStrategy userStrategy) const {
    auto targetTypes = getDistributionTypesWithStrategy(parentOp, parentStrategy, userOp, userStrategy);
    auto targetOutputType = targetTypes.first;
    auto targetInputType = targetTypes.second;
    return getSpillingCost(targetOutputType, targetInputType, parentOp, userOp);
}

VPU::MultiClusterStrategy LayerCostModel::getOptimalLayerStrategy(VPU::ClusteredOpInterface clusteredOp) {
    double splitOverHeightCost = COST_MAX;
    double splitOverKernelCost = COST_MAX;
    auto splitOverHeightFitIntoCMX = false;
    auto splitOverKernelFitIntoCMX = false;
    auto splitOverHeightIsCompatible = false;
    auto splitOverKernelIsCompatible = false;
    if (clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverHeight, _numTiles) &&
        clusteredOp.isOperationSplitOverHeightCompatible(/*vpux::TileInfo=*/vpux::TileInfo(ShapeRef()))) {
        splitOverHeightIsCompatible = true;
        splitOverHeightCost = getLayerCost(clusteredOp.getOperation(), VPU::MultiClusterStrategy::SplitOverHeight);
        _log.nest().trace("SplitOverHeight cost is {0}", splitOverHeightCost);
        splitOverHeightFitIntoCMX =
                clusteredOp.doesLayerFitIntoCMX(VPU::MultiClusterStrategy::SplitOverHeight, /*reservedMem=*/Byte(0));
    }

    if (clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::SplitOverKernel, _numTiles) &&
        clusteredOp.isOperationSplitOverKernelCompatible(/*outputShape=*/ShapeRef(), /*offset=*/ShapeRef(),
                                                         /*axis=*/ShapeRef())) {
        splitOverKernelIsCompatible = true;
        splitOverKernelCost = getLayerCost(clusteredOp.getOperation(), VPU::MultiClusterStrategy::SplitOverKernel);
        _log.nest().trace("SplitOverKernel cost is {0}", splitOverKernelCost);
        splitOverKernelFitIntoCMX =
                clusteredOp.doesLayerFitIntoCMX(VPU::MultiClusterStrategy::SplitOverKernel, /*reservedMem=*/Byte(0));
    }

    const auto optimalHeightTiling = [&](void) {
        return (mlir::isa<vpux::VPU::NCECompressConvolutionOp, vpux::VPU::NCEPermuteOp>(clusteredOp))
                       ? VPU::MultiClusterStrategy::SplitOverHeightOverlapped
                       : VPU::MultiClusterStrategy::SplitOverHeight;
    };

    // Check if SplitOverHeight is the only strategy which fits into CMX
    if (splitOverHeightFitIntoCMX && (!splitOverKernelFitIntoCMX)) {
        return optimalHeightTiling();
    }

    // Compute amount of clusters so that SOK is compatible
    const auto outputChannels =
            clusteredOp->getResult(0).getType().template cast<vpux::NDTypeInterface>().getShape()[Dims4D::Act::C];
    auto uniformDistributedSegments = VPU::isUniformDistributedSegmentsSupported(clusteredOp);
    const auto sokOptimalClusters =
            getNumberOfClustersForSOKToAvoidAlignment(outputChannels, _numTiles, uniformDistributedSegments);

    // Check if SplitOverKernel is the only strategy which fits into CMX and utilize full clusters
    if ((!splitOverHeightFitIntoCMX) && splitOverKernelFitIntoCMX && (sokOptimalClusters == _numTiles)) {
        return VPU::MultiClusterStrategy::SplitOverKernel;
    }

    if ((splitOverHeightCost != COST_MAX) && (splitOverKernelCost != COST_MAX) && (sokOptimalClusters == _numTiles)) {
        if (!hasUserMVN(clusteredOp) && splitOverHeightCost <= splitOverKernelCost) {
            return optimalHeightTiling();
        }
        return VPU::MultiClusterStrategy::SplitOverKernel;
    }
    // SOH is P1 option as SOK may not utilize full clusters
    // However, it is still more optimal than clustering
    if (splitOverHeightCost != COST_MAX) {
        return optimalHeightTiling();
    }

    if (splitOverKernelCost != COST_MAX) {
        return VPU::MultiClusterStrategy::SplitOverKernel;
    }

    // Handle the case when VPUNN is failing to return valid strategy cost due to unsupported parameters
    // Return SOH or SOK if any one is compatible, it's more optimal than clustering
    if (splitOverHeightIsCompatible) {
        return VPU::MultiClusterStrategy::SplitOverHeight;
    }

    if (splitOverKernelIsCompatible) {
        return VPU::MultiClusterStrategy::SplitOverKernel;
    }

    return VPU::MultiClusterStrategy::Clustering;
}

bool vpux::VPU::isStrategySOXCompatible(VPU::ClusteredOpInterface clusteredOp, VPU::MultiClusterStrategy strategy,
                                        size_t numTiles) {
    if (clusteredOp == nullptr) {
        return false;
    }

    if (clusteredOp.checkStrategyCompatibility(strategy, numTiles)) {
        switch (strategy) {
        case VPU::MultiClusterStrategy::SplitOverHeight:
        case VPU::MultiClusterStrategy::SplitOverHeightOverlapped:
            return clusteredOp.isOperationSplitOverHeightCompatible(/*vpux::TileInfo=*/vpux::TileInfo(ShapeRef()));
        case VPU::MultiClusterStrategy::SplitOverKernel:
            return clusteredOp.isOperationSplitOverKernelCompatible(/*outputShape=*/ShapeRef(), /*offset=*/ShapeRef(),
                                                                    /*axis=*/ShapeRef());
        case VPU::MultiClusterStrategy::SplitOverWidth:
            return clusteredOp.isOperationSplitOverWidthCompatible(/*outputShape=*/ShapeRef(), /*offset=*/ShapeRef(),
                                                                   /*axis=*/ShapeRef());
        default:
            return false;
        }
    }

    return false;
}

// For a clustered op which doesn't support cycle cost calculation the priority for strategies is parent strategy >
// SOH/SOHOverlapped > SOK > SOW > Clustering
std::optional<VPU::MultiClusterStrategy> vpux::VPU::getDefaultLayerStrategy(VPU::ClusteredOpInterface clusteredOp) {
    auto module = clusteredOp->getParentOfType<mlir::ModuleOp>();
    auto tileOp = IE::getTileExecutor(module);
    const auto numTiles = tileOp.getCount();

    // Try parent's strategy first
    auto parent = clusteredOp->getOperand(0);
    if (parent != nullptr) {
        if (auto parentClusterOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(parent.getDefiningOp())) {
            auto strategyAttr = parentClusterOp.getMultiClusterStrategy();
            if (strategyAttr.has_value()) {
                auto strategy = strategyAttr.value();
                if (isStrategySOXCompatible(clusteredOp, strategy, numTiles)) {
                    return strategy;
                }
            }
        }
    }

    // Only the highest dimension is prioritized
    // Need to investigate if the complete layout order could be optimal
    // Track E#124146
    auto strategyOrder = SmallVector(
            {VPU::MultiClusterStrategy::SplitOverHeight, VPU::MultiClusterStrategy::SplitOverHeightOverlapped,
             VPU::MultiClusterStrategy::SplitOverKernel, VPU::MultiClusterStrategy::SplitOverWidth});
    const auto outputType = clusteredOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto highestDim = getHighestDim(outputType.getShape(), outputType.getDimsOrder());
    if (highestDim == Dims4D::Act::C) {
        strategyOrder = SmallVector(
                {VPU::MultiClusterStrategy::SplitOverKernel, VPU::MultiClusterStrategy::SplitOverHeight,
                 VPU::MultiClusterStrategy::SplitOverHeightOverlapped, VPU::MultiClusterStrategy::SplitOverWidth});
    }

    for (auto strategy : strategyOrder) {
        if (!clusteredOp.checkStrategyCompatibility(strategy, numTiles)) {
            continue;
        }
        if (strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
            strategy == VPU::MultiClusterStrategy::SplitOverHeightOverlapped) {
            if (clusteredOp.isOperationSplitOverHeightCompatible(/*vpux::TileInfo=*/vpux::TileInfo(ShapeRef()))) {
                return strategy;
            }
        }
        if (strategy == VPU::MultiClusterStrategy::SplitOverKernel) {
            if (clusteredOp.isOperationSplitOverKernelCompatible(/*outputShape=*/ShapeRef(), /*offset=*/ShapeRef(),
                                                                 /*axis=*/ShapeRef())) {
                return strategy;
            }
        }
        if (strategy == VPU::MultiClusterStrategy::SplitOverWidth) {
            if (mlir::isa<VPU::SoftMaxOp, VPU::DepthToSpaceOp, VPU::PadOp, VPU::MVN1NormalizeOp, VPU::SwishOp>(
                        clusteredOp.getOperation()) &&
                clusteredOp.isOperationSplitOverWidthCompatible(/*outputShape=*/ShapeRef(), /*offset=*/ShapeRef(),
                                                                /*axis=*/ShapeRef())) {
                return strategy;
            }
        }
    }

    if (clusteredOp.checkStrategyCompatibility(VPU::MultiClusterStrategy::Clustering, numTiles)) {
        return VPU::MultiClusterStrategy::Clustering;
    }

    return std::nullopt;
}

bool vpux::VPU::isStrategyCompatibleShape(VPU::ClusteredOpInterface clusteredOp, const vpux::TileInfo& outputTile,
                                          VPU::MultiClusterStrategy strategy, Logger log) {
    auto shape = ShapeRef(outputTile.shape);

    if (shape.size() != RANK_REQUIRED_FOR_TILING && shape.size() != DimsGroups5D::Act::numDims) {
        log.trace("Operation '{0}' at '{1}' has output rank {2} and cannot be tiled. Expected rank: {3} or {4}.",
                  clusteredOp->getName(), clusteredOp->getLoc(), shape.size(), RANK_REQUIRED_FOR_TILING,
                  DimsGroups5D::Act::numDims);
        return false;
    }
    switch (strategy) {
    case MultiClusterStrategy::SplitOverHeight:
    case MultiClusterStrategy::SplitOverHeightOverlapped:
    case MultiClusterStrategy::HKSwitch: {
        return clusteredOp.isOperationSplitOverHeightCompatible(outputTile);
    }
    case MultiClusterStrategy::SplitOverHeightKernel: {
        return clusteredOp.isOperationSplitOverHeightCompatible(outputTile) &&
               clusteredOp.isOperationSplitOverKernelCompatible(outputTile.shape, outputTile.offsets, outputTile.axis);
    }
    case MultiClusterStrategy::SplitOverWidth: {
        return clusteredOp.isOperationSplitOverWidthCompatible(outputTile.shape, outputTile.offsets, outputTile.axis);
    }
    case MultiClusterStrategy::SplitOverKernel: {
        return clusteredOp.isOperationSplitOverKernelCompatible(outputTile.shape, outputTile.offsets, outputTile.axis);
    }
    case MultiClusterStrategy::SplitOverHeightWidth: {
        return clusteredOp.isOperationSplitOverHeightCompatible(outputTile) &&
               clusteredOp.isOperationSplitOverWidthCompatible(outputTile.shape, outputTile.offsets, outputTile.axis);
    }
    case MultiClusterStrategy::SplitOverBatch: {
        return clusteredOp.isOperationSplitOverBatchCompatible(outputTile.shape);
    }
    case MultiClusterStrategy::Clustering: {
        return true;
    }
    case MultiClusterStrategy::SplitOverGroup: {
        return clusteredOp.isOperationSplitOverGroupCompatible(outputTile);
    }
    default: {
        VPUX_THROW("Unknown multi cluster strategy {0}", strategy);
    }
    }
}

SmallVector<uint32_t> vpux::VPU::getDPUCostForNCEOp(VPU::NCEOpInterface nceOp, VPU::MultiClusterStrategy mcStrategy,
                                                    const OutputTiling& outTiles,
                                                    SmallVector<SmallVector<NDTypeInterface>>& tilesTypes,
                                                    const VPUIP::WorkloadCostParams& costParams,
                                                    VPUNN::VPULayerStrategy vpunnStrategy,
                                                    const std::shared_ptr<VPUNN::VPULayerCostModel>& vpunnCostModel,
                                                    Logger log) {
    std::vector<VPUNN::DPULayer> vpunnLayers{VPU::getDPULayer(costParams)};
    if (!outTiles.empty()) {
        auto tilingBuilderOp = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(nceOp.getOperation());
        VPUX_THROW_WHEN(tilingBuilderOp == nullptr, "NCE op {0} at {1} should be a tiling op", nceOp->getName(),
                        nceOp.getLoc());
        const auto tilingVPUNNLayer = [&](const VPUNN::DPULayer& vpunnLayer,
                                          const OutputTiling& outTiles) -> std::vector<VPUNN::DPULayer> {
            std::vector<VPUNN::DPULayer> vpunnLayers;
            vpunnLayers.reserve(outTiles.size());
            for (auto& outTile : outTiles) {
                vpunnLayers.push_back(vpunnLayer);
                auto inTiles = tilingBuilderOp.backInferTileInfo(outTile, log);
                tilesTypes.push_back(getTileTypes(nceOp.getOperation(), outTile, inTiles));
                auto& inputTile = inTiles.tiles.front();
                auto inPad = inTiles.pads;
                vpunnLayers.back().inputs = {getVPUTensor(inputTile.shape, costParams.inDataType)};
                vpunnLayers.back().outputs = {getVPUTensor(outTile.shape, costParams.outDataType)};
                if (inPad.has_value()) {
                    vpunnLayers.back().padding = {
                            static_cast<unsigned int>(inPad->top), static_cast<unsigned int>(inPad->bottom),
                            static_cast<unsigned int>(inPad->left), static_cast<unsigned int>(inPad->right)};
                }
            }
            return vpunnLayers;
        };
        vpunnLayers = tilingVPUNNLayer(vpunnLayers[0], outTiles);
    }

    // E#113592 For not supported SEP layer costs - optimize activation spills
    if (auto inputSparseTensorOp = nceOp->getOperand(0).getDefiningOp<VPU::GroupSparseTensorOp>()) {
        if (inputSparseTensorOp.getStorageElementTable() != nullptr) {
            return SmallVector<uint32_t>(vpunnLayers.size(), 1);
        }
    }

    SmallVector<uint32_t> layerDPUCosts;
    for (auto& vpunnLayer : vpunnLayers) {
        auto cost = checkAndReturnCost(vpunnCostModel->Layer(vpunnLayer, vpunnStrategy), log);
        if (cost >= VPU::INVALID_COST_BASE) {
            printVPUNNLayerConfig(vpunnLayer, vpunnStrategy, log);
            if (cost == VPU::ERROR_INPUT_TOO_BIG && !layerDPUCosts.empty()) {
                log.trace(" Use the first availabe layer cost to estimate the layer with ERROR_INPUT_TOO_BIG");
                cost = layerDPUCosts.front();
            } else {
                layerDPUCosts.clear();
                break;
            }
        }

        if (mlir::isa<VPU::NCEEltwiseOp>(nceOp.getOperation()) &&
            (mcStrategy == VPU::MultiClusterStrategy::Clustering)) {
            // The VPUNN cost of NCEEltwiseOp is inaccurate
            // Multiply a ratio to correct the cost
            // Track [E#98656]
            log.trace("Using NCEELTWISE_DPU_COST_RATIO for DPU cost");
            cost *= NCEELTWISE_DPU_COST_RATIO;
        }

        auto clusteredOp = mlir::cast<VPU::ClusteredOpInterface>(nceOp.getOperation());
        if (mlir::isa<VPU::NCEDepthConvolutionOp>(nceOp.getOperation()) &&
            (mcStrategy == VPU::MultiClusterStrategy::SplitOverKernel) &&
            (!VPU::isArchVPUX3XXX(VPU::getArch(clusteredOp)))) {
            auto modeIn = VPU::getActivationTensorDistributionMode(clusteredOp, mcStrategy);
            auto modeOut = VPU::getOutputTensorDistributionMode(clusteredOp, mcStrategy);

            // DUP -> DUP case
            if ((VPU::bitEnumContainsAny(modeIn, VPU::DistributionMode::DUPLICATED) ||
                 VPU::bitEnumContainsAny(modeIn, VPU::DistributionMode::MULTICASTED)) &&
                ((VPU::bitEnumContainsAny(modeOut, VPU::DistributionMode::DUPLICATED) ||
                  VPU::bitEnumContainsAny(modeOut, VPU::DistributionMode::MULTICASTED)))) {
                auto nTiles = vpunnStrategy.nTiles;
                VPUX_THROW_WHEN(nTiles <= 0, "nTiles should be positive but got {0}", nTiles);
                auto outputChannels = clusteredOp.getOperation()
                                              ->getResult(0)
                                              .getType()
                                              .cast<NDTypeInterface>()
                                              .getShape()[Dims4D::Act::C];
                size_t perTileCh = outputChannels / nTiles;
                if (perTileCh <= 32) {
                    // The VPUNN cost of NCEDWCONV is inaccurate
                    // Multiply a ratio to correct the cost
                    // Track [E#117314]
                    log.trace("Using NCEDWCONV_DPU_COST_RATIO for DPU cost");
                    cost *= NCEDWCONV_DPU_COST_RATIO;
                }
            }
        }

        if ((nceOp->getOperand(0).getType().dyn_cast<VPU::SparseTensorType>() ||
             nceOp->getResult(0).getType().dyn_cast<VPU::SparseTensorType>()) &&
            (mcStrategy == VPU::MultiClusterStrategy::SplitOverKernel)) {
            // The VPUNN cost of ACT-SPARSITY is inaccurate
            // Multiply a ratio to correct the cost
            // Track [E#117195]
            log.trace("Using ACTSPARSE_DPU_COST_RATIO for DPU cost");
            cost *= ACTSPARSE_DPU_COST_RATIO;
        }

        layerDPUCosts.push_back(cost);
    }

    return layerDPUCosts;
}

SmallVector<uint32_t> vpux::VPU::getPerTileWeightsDMACosts(
        VPU::NCEOpInterface nceOp, ArrayRef<SmallVector<NDTypeInterface>> tilesTypes,
        std::function<uint32_t(NDTypeInterface)> getSpillingReadCostFunc) {
    auto weightsOperand = nceOp.getWeightsOperand();
    if (weightsOperand == nullptr) {
        return SmallVector<uint32_t>(std::max<size_t>(tilesTypes.size(), 1), 0);
    }

    const auto inferredTileTypes = SmallVector<SmallVector<NDTypeInterface>>{
            getTileTypes(nceOp.getOperation(), TileInfo(getShape(nceOp->getResult(0))))};
    const auto& typesList = tilesTypes.empty() ? inferredTileTypes : tilesTypes;

    SmallVector<uint32_t> perTileWeightsCosts;
    for (const auto& tileTypes : typesList) {
        VPUX_THROW_UNLESS(tileTypes.size() > 1,
                          "NCEOp {0} at {1} has invalid number of tile types, got {2}, expected >1", nceOp->getName(),
                          nceOp->getLoc(), tileTypes.size());
        auto weightsDMACost = checked_cast<uint32_t>(getSpillingReadCostFunc(tileTypes[1]));
        perTileWeightsCosts.push_back(weightsDMACost);
    }

    return perTileWeightsCosts;
}

SmallVector<uint32_t> vpux::VPU::getPerTileActivationDMACosts(
        VPU::NCEOpInterface nceOp, ArrayRef<SmallVector<NDTypeInterface>> tilesTypes,
        std::function<uint32_t(NDTypeInterface)> getSpillingReadCostFunc) {
    auto getParentOp = [&]() {
        mlir::Operation* parentOp = nceOp->getOperand(0).getDefiningOp();
        while (parentOp && (mlir::isa<VPU::GroupSparseTensorOp>(parentOp) || isPureViewOp(parentOp))) {
            parentOp = parentOp->getOperand(0).getDefiningOp();
        }

        return parentOp;
    };

    // If op fit into CMX and parent op exists, we assume act spilling can be removed by adjusting startegy
    if ((tilesTypes.size() <= 1) && (getParentOp() != nullptr)) {
        return SmallVector<uint32_t>(std::max<size_t>(tilesTypes.size(), 1), 0);
    }

    bool isEltwiseOpWithDiffInputs =
            (mlir::isa<VPU::NCEEltwiseOp>(nceOp) && nceOp->getOperand(0) != nceOp->getOperand(1));

    SmallVector<uint32_t> perTileActCosts;
    for (const auto& tileTypes : tilesTypes) {
        VPUX_THROW_UNLESS(tileTypes.size() > 1,
                          "NCEOp {0} at {1} has invalid number of tile types, got {2}, expected >1", nceOp->getName(),
                          nceOp->getLoc(), tileTypes.size());
        auto actDMACost = checked_cast<uint32_t>(getSpillingReadCostFunc(tileTypes[0]));
        if (isEltwiseOpWithDiffInputs) {
            actDMACost += checked_cast<uint32_t>(getSpillingReadCostFunc(tileTypes[1]));
        }
        perTileActCosts.push_back(actDMACost);
    }

    return perTileActCosts;
}

SmallVector<uint32_t> vpux::VPU::getPerTileOutputDMACosts(
        VPU::NCEOpInterface nceOp, ArrayRef<SmallVector<NDTypeInterface>> tilesTypes,
        std::function<uint32_t(NDTypeInterface)> getSpillingWriteCostFunc) {
    SmallVector<uint32_t> perTileOutputCosts;
    for (const auto& tileTypes : tilesTypes) {
        VPUX_THROW_UNLESS(tileTypes.size() > 1,
                          "NCEOp {0} at {1} has invalid number of tile types, got {2}, expected > 1", nceOp->getName(),
                          nceOp->getLoc(), tileTypes.size());
        auto outputDMACost = checked_cast<uint32_t>(getSpillingWriteCostFunc(tileTypes.back()));
        perTileOutputCosts.push_back(outputDMACost);
    }

    return perTileOutputCosts;
}

uint32_t vpux::VPU::getWeightsDMACostForNCEOp(VPU::NCEOpInterface nceOp, const OutputTiling& outTiles,
                                              SmallVector<uint32_t>& layerDPUCosts, ArrayRef<uint32_t> layerDMACosts,
                                              uint32_t parentLastDPUCostToOverlap, bool enablePrefetchTiling,
                                              vpux::Logger log) {
    VPUX_THROW_WHEN(layerDPUCosts.empty() || layerDPUCosts.size() != layerDMACosts.size(),
                    "Layer DPU costs must be non-empty and equal to DMA costs in size");

    const auto outShape = getShape(nceOp->getResult(0));
    auto tiles = outTiles.empty() ? OutputTiling({TileInfo(outShape)}) : outTiles;

    auto weightsOperand = nceOp.getWeightsOperand();
    bool isWeightsDMASplitOnEachTile = (weightsOperand != nullptr && tiles.front().axis[Dims4D::Act::C] > 1);

    auto tilingInfoOp = mlir::dyn_cast<VPU::TilingInfoOpInterface>(nceOp.getOperation());
    // If the DMA will overlap with DPU from the second tile on
    bool isDMAOverlappedWithDPU =
            enablePrefetchTiling ? tilingInfoOp != nullptr &&
                                           tilingInfoOp.isSupportedTiling(tiles, vpux::TilingMode::PIPELINING, log)
                                 : false;

    uint32_t totalDMACost = 0;

    if (isDMAOverlappedWithDPU) {
        // Weights DMA from second tile on will be overlapped with DPU of previous tile
        totalDMACost +=
                getPrefetchDMACostOverlappsWithPreviousDPU(layerDPUCosts, layerDMACosts, isWeightsDMASplitOnEachTile);
    } else {
        // When DMA not overlapped with DPU
        //  - If weights DMA will be copied on each tile, we need to accumulate all the DMA costs
        //  - If weights DMA will be shared for all tiles, we only add the first DMA cost
        totalDMACost += isWeightsDMASplitOnEachTile ? std::accumulate(layerDMACosts.begin(), layerDMACosts.end(), 0U)
                                                    : layerDMACosts.front();
    }

    // the first DMA will overlap with previous op's DPU, so exclude the overlapped part from cost
    totalDMACost -= std::min(parentLastDPUCostToOverlap, layerDMACosts.front());

    return totalDMACost;
}

uint32_t vpux::VPU::getActivationDMACostForNCEOp(VPU::NCEOpInterface nceOp, const OutputTiling& outTiles,
                                                 SmallVector<uint32_t>& layerDPUCosts, ArrayRef<uint32_t> layerDMACosts,
                                                 bool enablePrefetchTiling, vpux::Logger log) {
    VPUX_THROW_WHEN(layerDPUCosts.empty() || layerDPUCosts.size() != layerDMACosts.size(),
                    "Layer DPU costs must be non-empty and equal to DMA costs in size");

    const auto outShape = getShape(nceOp->getResult(0));
    auto tiles = outTiles.empty() ? OutputTiling({TileInfo(outShape)}) : outTiles;

    auto weightsOperand = nceOp.getWeightsOperand();
    // If the activation needs to be split:
    //      no weights operand - like Eltwise
    //      tiling on spatial dimension
    //      or DepthConvolution, the activation input is always split for tile, and the DMAs should be accumulated
    bool isActDMASplitOnEachTile = (weightsOperand == nullptr || tiles.front().axis[Dims4D::Act::C] == 1 ||
                                    mlir::isa<VPU::NCEDepthConvolutionOp>(nceOp.getOperation()));

    auto tilingInfoOp = mlir::dyn_cast<VPU::TilingInfoOpInterface>(nceOp.getOperation());
    // The DMA will overlap with DPU from the second tile on
    bool isDMAOverlappedWithDPU =
            enablePrefetchTiling ? tilingInfoOp != nullptr &&
                                           tilingInfoOp.isSupportedTiling(tiles, vpux::TilingMode::PIPELINING, log)
                                 : false;

    uint32_t totalDMACost = 0;

    if (isDMAOverlappedWithDPU) {
        // Act DMA from second tile on will be overlapped with DPU of previous tile
        totalDMACost +=
                getPrefetchDMACostOverlappsWithPreviousDPU(layerDPUCosts, layerDMACosts, isActDMASplitOnEachTile);
    } else {
        // When DMA not overlapped with DPU
        //  - If act DMA will be copied on each tile, we need to accumulate all the DMA costs
        //  - If act DMA will be shared for all tiles, we only add the first DMA cost
        totalDMACost += isActDMASplitOnEachTile ? std::accumulate(layerDMACosts.begin(), layerDMACosts.end(), 0U)
                                                : layerDMACosts.front();
    }

    return totalDMACost;
}

uint32_t vpux::VPU::getOutputDMACostForNCEOp(VPU::NCEOpInterface nceOp, const OutputTiling& outTiles,
                                             SmallVector<uint32_t>& layerDPUCosts, ArrayRef<uint32_t> layerDMACosts,
                                             bool enablePrefetchTiling, vpux::Logger log) {
    VPUX_THROW_WHEN(layerDPUCosts.empty() || layerDPUCosts.size() != layerDMACosts.size(),
                    "Layer DPU costs must be non-empty and equal to DMA costs in size");

    const auto outShape = getShape(nceOp->getResult(0));
    auto tiles = outTiles.empty() ? OutputTiling({TileInfo(outShape)}) : outTiles;

    auto tilingInfoOp = mlir::dyn_cast<VPU::TilingInfoOpInterface>(nceOp.getOperation());
    // The DMA of the current tile will overlap with DPU of the next tile
    nceOp->setAttr(outputPipelining, mlir::BoolAttr::get(nceOp->getContext(), true));
    bool isDMAOverlappedWithDPU =
            enablePrefetchTiling ? tilingInfoOp != nullptr &&
                                           tilingInfoOp.isSupportedTiling(tiles, vpux::TilingMode::PIPELINING, log)
                                 : false;
    nceOp->removeAttr(outputPipelining);

    uint32_t totalDMACost = 0;

    if (isDMAOverlappedWithDPU) {
        // Output DMA expect for the last tile will be overlapped with DPU of the next tile
        totalDMACost += getOutputDMACostOverlappsWithNextDPU(layerDPUCosts, layerDMACosts, true);
    } else {
        totalDMACost += std::accumulate(layerDMACosts.begin(), layerDMACosts.end(), 0U);
    }

    return totalDMACost;
}

size_t vpux::VPU::getNumNonConstantOperands(mlir::Operation* op) {
    return std::count_if(op->operand_begin(), op->operand_end(), [](mlir::Value operand) {
        return !mlir::isa_and_nonnull<Const::DeclareOp>(operand.getDefiningOp());
    });
}

bool vpux::VPU::hasLayerWithMultipleInputs(mlir::Operation* op) {
    return std::any_of(op->user_begin(), op->user_end(), [](mlir::Operation* user) {
        return getNumNonConstantOperands(user) > 1 || hasLayerWithMultipleInputs(user);
    });
}

bool vpux::VPU::isSingleBatchRequired(mlir::Operation* op) {
    return !mlir::isa<VPU::MVN1NormalizeOp, VPU::MVN1SumOp>(op);
}
